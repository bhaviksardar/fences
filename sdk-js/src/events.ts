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

const warned = new Set<string>();
/** Log a warning once per key per process. */
export function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`agentfences: ${message}`);
}

export function configure(environment?: string, release?: string, redact?: RedactHook): void {
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

/**
 * Pass an outgoing event through the user's redact hook. Returns the event to send, or
 * null to drop it. If the hook throws or returns something else, the event is dropped:
 * never send unredacted data, and never crash the agent.
 */
export function redact(event: Event): Event | null {
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
