/**
 * What a run carries besides numbers: its context (environment, release, user, session),
 * the error that ended it, and the redact hook every outgoing event passes through.
 */
import { AsyncLocalStorage } from "node:async_hooks";
import { FencesAuthError, FencesError } from "./errors.js";

export const MAX_CONTEXT_KEYS = 32;
export const MAX_CONTEXT_VALUE = 256;
export const MAX_MESSAGE = 1000;
export const MAX_FRAMES = 15;

export type Event = { type: string; [key: string]: unknown };
export type RedactHook = (event: Event) => Event | null | undefined;

const processContext: Record<string, string> = {};  // environment / release, from init()
const contextStore = new AsyncLocalStorage<Record<string, string>>();
let redactHook: RedactHook | undefined;
let redactFailed = false;
export type Capture = "full" | "metadata";
/** init({ capture }): "metadata" sends nothing the agent or its users wrote. */
export let capture: Capture = "full";

const warned = new Set<string>();
/** Log a warning once per key per process. */
export function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`agentfences: ${message}`);
}

export function configure(environment?: string, release?: string, redact?: RedactHook, capture_: unknown = "full"): void {
  if (capture_ !== "full" && capture_ !== "metadata") throw new TypeError(`capture must be "full" or "metadata", not ${JSON.stringify(capture_)}`);
  capture = capture_;
  for (const k of Object.keys(processContext)) delete processContext[k];
  if (environment) processContext.environment = environment.slice(0, MAX_CONTEXT_VALUE);
  if (release) processContext.release = release.slice(0, MAX_CONTEXT_VALUE);
  redactHook = redact;
  redactFailed = false;
}

/**
 * Tag every run started inside fn, e.g. with the end user or session it serves:
 *
 *     await context({ user_id: "u_123", session_id: "s_9" }, () => supportAgent(question))
 *
 * Calls nest; inner values win. Values are stored as strings (up to 256 characters,
 * 32 keys); null and undefined leave a key out.
 */
export function context<T>(values: Record<string, unknown>, fn: () => T): T {
  const merged = { ...(contextStore.getStore() ?? {}) };
  for (const [k, v] of Object.entries(values)) {
    if (v !== null && v !== undefined) merged[k] = String(v).slice(0, MAX_CONTEXT_VALUE);
  }
  if (Object.keys(merged).length > MAX_CONTEXT_KEYS) {
    throw new RangeError(`agentfences.context() takes at most ${MAX_CONTEXT_KEYS} keys`);
  }
  return contextStore.run(merged, fn);
}

export function currentContext(): Record<string, string> {
  return { ...processContext, ...(contextStore.getStore() ?? {}) };
}

export const MAX_TAGS = 20;
export const MAX_TAG_VALUE = 100;
const TAG_KEY = /^[a-z0-9_.-]{1,40}$/;

/** init({ tags }) and governed(fn, { tags }): up to 20 keys of a-z, 0-9, _ . - (40 at most), string values of 100 at most. */
export function checkTags(tags: unknown): Record<string, string> {
  if (tags === undefined || tags === null) return {};
  if (typeof tags !== "object" || Array.isArray(tags)) throw new TypeError(`tags must be an object, not ${JSON.stringify(tags)}`);
  const entries = Object.entries(tags);
  if (entries.length > MAX_TAGS) throw new RangeError(`tags takes at most ${MAX_TAGS} keys, not ${entries.length}`);
  for (const [k, v] of entries) {
    if (!TAG_KEY.test(k)) throw new RangeError(`tag key ${JSON.stringify(k)} must be 1 to 40 of a-z, 0-9, _ . -`);
    if (typeof v !== "string" || v.length > MAX_TAG_VALUE) {
      throw new RangeError(`tag ${JSON.stringify(k)} must be a string of at most ${MAX_TAG_VALUE} characters, not ${JSON.stringify(v)}`);
    }
  }
  return Object.fromEntries(entries) as Record<string, string>;
}

/**
 * The run start's context: its own keys plus each tag as `tag.<key>` (the run start has no field for
 * tags, and older servers may refuse new ones). Servers take 32 keys at most: past that, environment,
 * release and the tags are kept first, then as many context() values as fit, with a warning.
 */
export function withTags(context: Record<string, string>, tags: Record<string, string>): Record<string, string> {
  const tagged = Object.fromEntries(Object.entries(tags).map(([k, v]) => [`tag.${k}`, v]));
  const merged = { ...context, ...tagged };
  if (Object.keys(merged).length <= MAX_CONTEXT_KEYS) return merged;
  const kept: Record<string, string> = {};
  for (const k of ["environment", "release"]) if (k in context) kept[k] = context[k];
  Object.assign(kept, tagged);
  for (const [k, v] of Object.entries(context)) {
    if (Object.keys(kept).length >= MAX_CONTEXT_KEYS) break;
    if (!(k in kept)) kept[k] = v;
  }
  warnOnce("context-trimmed", `a run's context and tags come to ${Object.keys(merged).length} keys, more than the ` +
    `${MAX_CONTEXT_KEYS} Fences takes; sent ${Object.keys(kept).length}, its tags first`);
  return kept;
}

/** `Type: message`, or just the type when there's no message. */
export function errorText(e: unknown): string {
  const type = errorType(e);
  const message = e instanceof Error ? e.message : e === undefined ? "" : String(e);
  return (message ? `${type}: ${message}` : type).slice(0, MAX_MESSAGE);
}

export function errorType(e: unknown): string {
  if (e instanceof Error) return e.name && e.name !== "Error" ? e.name : e.constructor.name;
  return typeof e;
}

export type ErrorCategory = "rate_limit" | "overloaded" | "timeout" | "auth" | "bad_request" | "server" | "network"
  | "cancelled" | "blocked" | "tool_error" | "model_error" | "unknown";

// For errors without an HTTP status: words in the class's name or its bases', checked in this order
// (an APIConnectionTimeoutError is also an APIConnectionError: a timeout first)
const NAMED: [ErrorCategory, string[]][] = [
  ["timeout", ["Timeout", "TimedOut", "DeadlineExceeded"]],
  ["rate_limit", ["RateLimit", "TooManyRequests", "ResourceExhausted"]],
  ["overloaded", ["Overloaded", "ServiceUnavailable"]],
  ["auth", ["Authentication", "Unauthorized", "Unauthenticated", "PermissionDenied", "Forbidden"]],
  ["network", ["Connect", "Network"]],
  ["cancelled", ["Cancel", "Abort", "Halt"]],
  ["bad_request", ["BadRequest", "InvalidRequest", "InvalidArgument", "NotFound", "UnprocessableEntity"]],
  ["server", ["InternalServer", "ServerError"]],
];
// Node's own error codes, on the error or its cause (fetch rejects with TypeError("fetch failed"))
const TIMEOUT_CODES = new Set(["ETIMEDOUT", "UND_ERR_CONNECT_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT"]);
const NETWORK_CODES = new Set(["ECONNREFUSED", "ECONNRESET", "ENOTFOUND", "EAI_AGAIN", "EPIPE", "EHOSTUNREACH", "ENETUNREACH", "UND_ERR_SOCKET"]);

/** The HTTP status an error carries: status (OpenAI, Anthropic, Gemini), statusCode, a numeric code, response.status. */
export function httpStatus(e: unknown): number | undefined {
  const o = e as { status?: unknown; statusCode?: unknown; code?: unknown; response?: { status?: unknown } } | null;
  for (const v of [o?.status, o?.statusCode, o?.code, o?.response?.status]) {
    if (typeof v === "number" && Number.isInteger(v) && v >= 100 && v <= 599) return v;
  }
  return undefined;
}

/**
 * What kind of failure this is, from its HTTP status, class names and Node error codes (no
 * provider SDK is imported): rate_limit, overloaded, timeout, auth, bad_request, server, network,
 * cancelled, blocked (Fences refused it) or unknown. Spans of failed tool and model calls say
 * tool_error and model_error for what's unknown here.
 */
export function categorize(e: unknown): ErrorCategory {
  try {
    if (e instanceof FencesAuthError) return "auth";
    if (e instanceof FencesError) return "blocked";
    const status = httpStatus(e);
    if (status !== undefined && status >= 400) {
      return status === 429 ? "rate_limit" : status === 503 || status === 529 ? "overloaded" : status === 401 || status === 403 ? "auth"
        : status === 408 || status === 504 ? "timeout" : status < 500 ? "bad_request" : "server";
    }
    const codes = [(e as { code?: unknown })?.code, ((e as { cause?: { code?: unknown } })?.cause)?.code];
    if (codes.some((c) => TIMEOUT_CODES.has(c as string))) return "timeout";
    if (codes.some((c) => NETWORK_CODES.has(c as string))) return "network";
    const names: string[] = [];
    if (e instanceof Error) names.push(e.name);
    for (let p = e !== null && typeof e === "object" ? Object.getPrototypeOf(e) : null; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
      names.push(p.constructor?.name ?? "");
    }
    const joined = names.join(" ");
    return NAMED.find(([, words]) => words.some((w) => joined.includes(w)))?.[0] ?? "unknown";
  } catch {  // an odd error object (a getter that throws) must not break recording it
    return "unknown";
  }
}

export interface ExceptionInfo { type: string; message: string; stack: string[] }

/** Type, message and the last frames of the stack (innermost last), for the run's error report. */
export function exceptionInfo(e: unknown): ExceptionInfo {
  const frames = e instanceof Error && e.stack
    ? e.stack.split("\n").slice(1).map((l) => l.trim()).filter((l) => l.startsWith("at ")).map((l) => {
        const m = l.match(/^at (?:(.+?) \()?(.+?):(\d+):\d+\)?$/);  // "at fn (file:line:col)" or "at file:line:col"
        return m ? `${m[2]}:${m[3]} in ${m[1] ?? "<anonymous>"}` : l.slice(3);
      })
    : [];
  return {
    type: errorType(e),
    message: (e instanceof Error ? e.message : String(e)).slice(0, MAX_MESSAGE),
    stack: frames.reverse().slice(-MAX_FRAMES),
  };
}

// capture "metadata": span attributes that carry what the agent, its users or its tools wrote
const FREE_TEXT = ["fences.error.message", "gen_ai.tool.call.arguments", "fences.approval.reason", "fences.approval.note"];
const CATEGORIES = new Set<unknown>(["rate_limit", "overloaded", "timeout", "auth", "bad_request", "server", "network", "cancelled",
  "blocked", "tool_error", "model_error", "unknown"]);

/**
 * The event with nothing the agent or its users wrote: names, models, numbers, statuses and error
 * types stay; messages, reasoning, arguments and context values go. A new object, never the one
 * passed in (spans are also kept on the run). null: don't send it at all.
 */
export function metadataOnly(event: Event): Event | null {
  if (event.type === "decision") return null;  // sent as a fences.decision span with its length instead (logDecision)
  if (event.type === "run_start" && event.context) {  // the keys, not the values; environment, release and tags stay
    const context = Object.fromEntries(Object.entries(event.context as Record<string, string>)
      .map(([k, v]) => [k, k === "environment" || k === "release" || k.startsWith("tag.") ? v : ""]));
    return { ...event, context };
  }
  if (event.type === "run_end" && event.exception) {
    const exception = event.exception as ExceptionInfo;
    return { ...event, error: exception.type, exception: { ...exception, message: "" } };
  }
  if (event.type === "span") {
    const attributes = { ...(event.attributes as Record<string, unknown>) };
    for (const k of FREE_TEXT) delete attributes[k];
    // A category stays, a label of the caller's goes
    if ("fences.fallback.reason" in attributes && !CATEGORIES.has(attributes["fences.fallback.reason"])) delete attributes["fences.fallback.reason"];
    return { ...event, attributes };
  }
  return event;  // a guarded tool's check keeps its arguments: rules need them
}

/**
 * Pass an outgoing event through the metadata filter (capture "metadata"), then the user's redact
 * hook. Returns the event to send, or null to drop it. If the hook throws or returns something else,
 * the event is dropped: never send unredacted data, and never crash the agent. The hook only sees
 * what the filter kept, so it can't add back what was dropped.
 */
export function redact(event: Event): Event | null {
  if (capture === "metadata") {
    const kept = metadataOnly(event);
    if (!kept) return null;
    event = kept;
  }
  if (!redactHook) return event;
  try {
    const out = redactHook(structuredClone(event));
    if (out === null || out === undefined) return null;
    if (typeof out !== "object" || Array.isArray(out)) throw new TypeError(`returned ${typeof out}, not an object or null`);
    return out;
  } catch (e) {
    if (!redactFailed) {
      redactFailed = true;
      console.warn(`agentfences: redact hook failed (${errorText(e)}); dropping events it fails on`);
    }
    return null;
  }
}
