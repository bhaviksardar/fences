/**
 * init(), governed() and checkpoint(): starting and ending runs, checking every step against
 * its limits (the server decides in cloud mode, the same rules apply here otherwise), and
 * recording decisions and spans.
 */
import { randomUUID } from "node:crypto";
import { FencesClient, type Reply } from "./client.js";
import * as control from "./control.js";
import { AgentQuarantined, FencesStop } from "./errors.js";
import * as events from "./events.js";
import * as outbox from "./outbox.js";
import * as precheck from "./precheck.js";
import { costOf, setCustomPrices, type CustomPrice } from "./pricing.js";
import { setToolArguments, type ToolArguments } from "./spans.js";
import { breachResult, okResult, type CheckpointResult } from "./results.js";
import { DEFAULTS, MAX_EVENTS_PER_RUN, Run, getActiveRun, store, type Limits, type Span } from "./run.js";

export type { CheckpointResult } from "./results.js";
export { Run, getActiveRun, type Limits, type Span } from "./run.js";

// ── Setup ─────────────────────────────────────────────────────────────────────

export interface InitOptions {
  /** Your Fences API key (fc_...). Required unless localOnly. */
  apiKey?: string;
  /** Your Fences server. Defaults to http://localhost:8000. */
  endpoint?: string;
  /** Enforce limits in-process only: no account, nothing leaves the process. */
  localOnly?: boolean;
  /** Treat an unreachable server as a breach (fences_unreachable) instead of enforcing locally. */
  failClosed?: boolean;
  /** Add or override model prices, USD per 1M tokens. */
  prices?: Record<string, CustomPrice>;
  /** Tag every run, e.g. "prod". */
  environment?: string;
  /** Tag every run, e.g. "v1.4.2". */
  release?: string;
  /** Sees every outgoing event and returns it (changed or not), or null to drop it. */
  redact?: events.RedactHook;
  /** Seconds between heartbeats reporting live runs (cloud mode). Default 15. */
  heartbeatS?: number;
  /** How long a paused run waits at its checkpoint before stopping with breach "paused". Default 3600. */
  pauseTimeoutS?: number;
  /**
   * If a run doesn't reach a checkpoint within this many seconds of a dashboard stop (say, it's
   * stuck in a tool), the governed call rejects with FencesStop and the run's signal aborts.
   * Default 30; null never does.
   */
  stopGraceS?: number | null;
  /**
   * Run in the agent's process when a stop arrives from the dashboard (once per run), to
   * contain what Fences can't reach, e.g. revoke a credential the agent holds. Whether it
   * worked, threw or ran past 30 seconds is reported to Fences.
   */
  onStop?: control.Hook;
  /** The same, when the agent is quarantined (once per quarantine, however many runs it reaches). */
  onQuarantine?: control.Hook;
  /**
   * "hold" makes runs wait at a limit instead of ending: when the server says a limit is crossed,
   * checkpoint() waits (as if paused) until someone raises it and resumes the run, then returns ok
   * and the agent carries on. A guarded tool denied for budget waits the same way. A stop ends the
   * wait, and after pauseTimeoutS the breach is returned as usual. Cloud only; override per agent in governed().
   */
  onBreach?: "hold" | null;
  /**
   * What a recorded tool call carries of its arguments: "names" (the default: `query, limit`),
   * "full" (the arguments as JSON, capped) or "none". A guarded tool's check and an approval request
   * still send the values, which rules and approvers need; the redact hook can change or drop them.
   */
  toolArguments?: ToolArguments;
  /**
   * endpoint must use https://, except on this machine (localhost, 127.0.0.1, ::1): over plain HTTP anyone
   * on the network path could read the API key and rewrite the server's replies, limits included.
   * true permits it, for a trusted private network.
   */
  allowHttp?: boolean;
}

let client: FencesClient | null = null;
let localOnly = false;
let failClosed = false;
let defaultOnBreach: "hold" | null = null;

function checkOnBreach(value: unknown): "hold" | null {
  if (value === undefined || value === null) return null;
  if (value !== "hold") throw new TypeError(`onBreach must be "hold" or null, not ${JSON.stringify(value)}`);
  return value;
}

/** Call once at startup, before any governed function runs. */
export function init(options: InitOptions = {}): void {
  setToolArguments(options.toolArguments ?? "names");
  setCustomPrices(options.prices);
  events.configure(options.environment, options.release, options.redact);
  localOnly = !!options.localOnly;
  failClosed = !!options.failClosed;
  defaultOnBreach = checkOnBreach(options.onBreach);
  outbox.resetSpans();  // a new server gets asked again
  if (localOnly) {
    client = null;
  } else if (!options.apiKey) {
    throw new Error("apiKey is required unless localOnly is set. Use init({ localOnly: true }) for local usage.");
  } else {
    const endpoint = options.endpoint ?? "http://localhost:8000";
    checkEndpoint(endpoint, !!options.allowHttp);
    client = new FencesClient(options.apiKey, endpoint);
  }
  control.configure(client, { heartbeatS: options.heartbeatS ?? 15, pauseTimeoutS: options.pauseTimeoutS ?? 3600,
    stopGraceS: options.stopGraceS === undefined ? 30 : options.stopGraceS, onStop: options.onStop, onQuarantine: options.onQuarantine });
}

/** @internal For tests: point the SDK at a fake server client. */
export function _setClient(c: FencesClient | null): void {
  client = c;
  control.configure(c, { heartbeatS: 15, pauseTimeoutS: 3600, stopGraceS: 30 });
}

/** @internal */
export const _client = (): FencesClient | null => client;

function checkEndpoint(endpoint: string, allowHttp: boolean): void {
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new TypeError(`endpoint must be an https:// URL, not ${JSON.stringify(endpoint)}`);
  }
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password || endpoint.includes("\\")) {
    throw new TypeError(`endpoint must be an https:// URL with no username or backslash, not ${JSON.stringify(endpoint)}`);
  }
  if (url.protocol === "https:") return;
  const host = url.hostname;  // parsed as fetch will: the host checked is the host connected to
  const local = host === "localhost" || host === "127.0.0.1" || host === "[::1]";
  if (!local && !allowHttp) {
    throw new TypeError(`endpoint ${JSON.stringify(endpoint)} uses plain HTTP: the API key and the server's replies, limits ` +
      "included, would cross the network unencrypted. Use https://, or pass allowHttp: true for a trusted private network.");
  }
}

function requireInit(): void {
  if (!localOnly && !client) {
    throw new Error("Call agentfences.init() before using Fences. For local usage: init({ localOnly: true })");
  }
}

function warnUnreachable(run: Run, err: string, ending = false): void {
  if (run.warned) return;
  run.warned = true;
  console.warn(`agentfences: Fences backend unreachable for run ${run.runId} (${err}); ${ending ? "ending the run" : "enforcing limits locally"}`);
}

// ── governed ──────────────────────────────────────────────────────────────────

/**
 * Make each call of fn a governed run with these limits:
 *
 *     const research = governed(async (q: string) => { ... }, { budgetUsd: 0.5, maxIterations: 20 });
 *
 * Every limit is optional. On Fences Cloud, a limit left out comes from the agent's page in
 * the dashboard. The run is named after the function (or `name`). checkpoint() never throws:
 * a crossed limit comes back as a breached result, so the agent can wrap up gracefully.
 */
export function governed<A extends unknown[], R>(
  fn: (...args: A) => R | Promise<R>,
  options: Limits & { name?: string; onBreach?: "hold" | null } = {},
): (...args: A) => Promise<Awaited<R>> {
  const name = options.name ?? (fn.name || "agent");
  checkOnBreach(options.onBreach);
  return async (...args: A): Promise<Awaited<R>> => {
    requireInit();
    const run = await startRun(name, options);
    run.onBreach = checkOnBreach(options.onBreach) ?? defaultOnBreach;
    // A stop the agent doesn't reach a checkpoint for within stopGraceS halts the call here
    const halted = new Promise<never>((_, reject) => {
      run.control.onCancel = () => {
        const stop = new FencesStop(finalResult(run));
        run._abort(stop);
        reject(stop);
      };
    });
    control.register(run);
    let error: unknown, failed = false;
    try {
      // ponytail: JS can't stop running code; the agent's own work goes on unless it uses run.signal
      return await Promise.race([store.run(run, () => fn(...args)) as Promise<Awaited<R>>, halted]);
    } catch (e) {
      [error, failed] = [e, true];
      throw e;
    } finally {
      run.control.onCancel = null;  // first, so a late cancel can't land after the call is over
      await endRun(run, ...outcome(run, failed, error));
    }
  };
}

function outcome(run: Run, failed: boolean, error: unknown): [string, string | null] {
  if (!failed) return [run.lastBreach ? "breached" : "success", null];
  if (error instanceof FencesStop) return ["breached", null];  // halted after a stop it missed (stopGraceS)
  run.exception = events.exceptionInfo(error);
  const e = run.exception;
  return ["error", e.message ? `${e.type}: ${e.message}` : e.type];
}

async function startRun(agentName: string, limits: Limits): Promise<Run> {
  const run = new Run(agentName, limits.budgetUsd ?? null, limits.maxIterations ?? DEFAULTS.maxIterations,
    limits.maxDurationMs ?? DEFAULTS.maxDurationMs, limits.maxTokens ?? DEFAULTS.maxTokens, events.currentContext());
  run.budgetInCode = limits.budgetUsd != null;
  if (!client) {
    if (run.budgetUsd === null) noBudget(agentName, "Set one with governed(fn, { budgetUsd: ... }).");
    return run;
  }
  // The run must start even if redaction drops its context: limits depend on it
  const sent = Object.keys(run.context).length
    ? events.redact({ type: "run_start", agent_name: agentName, context: { ...run.context } }) : null;
  // Limits left out in code go as null: the server fills them from the agent's dashboard settings
  const body: Record<string, unknown> = { run_id: run.runId, agent_name: agentName, budget_usd: limits.budgetUsd ?? null,
    max_iterations: limits.maxIterations ?? null, max_duration_ms: limits.maxDurationMs ?? null, max_tokens: limits.maxTokens ?? null };
  if (sent?.context) body.context = sent.context;
  const resp = await client.startRun(body);
  if (resp.quarantined) throw new AgentQuarantined(agentName, resp.detail);
  // The server caps limits at the agent's ceilings: enforce the same numbers if it becomes unreachable later
  // A null budget (unlimited) is taken only for a run whose code set none: a reply can't lift a budget from the code
  const isObject = (v: unknown): v is Record<string, any> => v !== null && typeof v === "object" && !Array.isArray(v);
  control.adopt(run, isObject(resp.limits) ? resp.limits : {}, control.LIMITS, limits.budgetUsd == null);
  for (const n of (Array.isArray(resp.notices) ? resp.notices : []).filter(isObject)) {  // a malformed one is skipped
    events.warnOnce(`${agentName}:${n.code}`, `${n.message ?? ""}${n.url ? ` ${n.url}` : ""}`);
  }
  if (!("limits" in resp) && run.budgetUsd === null && !resp.network_error) {
    noBudget(agentName, "Set one on its page in the Fences dashboard.");
  }
  if (resp.network_error) warnUnreachable(run, resp.network_error);
  return run;
}

function noBudget(agentName: string, how: string): void {
  events.warnOnce(`${agentName}:no_budget`, `${agentName} has no budget, so its spend isn't capped. ${how}`);
}

async function endRun(run: Run, status: string, error: string | null): Promise<void> {
  control.unregister(run);
  if (!client) return;
  // The run must end even if redaction drops its error details
  const sent = error ? events.redact({ type: "run_end", status, error, exception: run.exception }) : null;
  const body: Record<string, unknown> = { status, error: sent?.error ?? null };
  if (sent?.exception) body.exception = sent.exception;
  if (status === "breached" && run.lastBreach) body.breach = run.lastBreach;  // matters for breaches the SDK decided
  await client.endRun(run.runId, body);
}

// ── checkpoint ────────────────────────────────────────────────────────────────

export interface Extra {
  /** Extra spend for this step in USD, e.g. a paid tool call. Without a response, the step's whole cost. */
  costDeltaUsd?: number;
  /** Extra tokens for this step. */
  tokensUsed?: number;
}

/**
 * Record one step, then check every limit. Pass the model's response and Fences works out
 * the cost from its model and token usage:
 *
 *     const response = await openai.chat.completions.create({ ... });
 *     const result = await checkpoint(response);
 *     if (result.breached) return result.message;
 *
 * Without a response: checkpoint(null, { costDeltaUsd: 0.02, tokensUsed: 450 }).
 * Never throws for a breach. Outside a governed run it does nothing.
 */
export async function checkpoint(response?: unknown, extra: Extra = {}): Promise<CheckpointResult> {
  const run = getActiveRun();
  if (!run) return okResult();
  if (typeof response === "number") {
    throw new TypeError("checkpoint() takes the model's response first; pass a cost as checkpoint(null, { costDeltaUsd })");
  }
  const [cost, tokens] = response === undefined || response === null ? [0, 0] : costOf(response);
  const stepCost = cost + checkAmount(extra.costDeltaUsd ?? 0, "costDeltaUsd");
  const stepTokens = tokens + checkAmount(extra.tokensUsed ?? 0, "tokensUsed");
  run.costUsd += stepCost;  // the actual cost replaces the hold of the call it prices
  precheck.releaseFinished(run, response ?? undefined);
  run.tokensUsed += stepTokens;
  run.iterations += 1;
  // A step_id per step: a retried checkpoint is counted once on servers that support it
  const resp = client ? await client.checkpoint(run.runId, { cost_delta_usd: stepCost, tokens_used: stepTokens,
    iterations: run.iterations, duration_ms: run.durationMs, step_id: randomUUID().replaceAll("-", "") }) : null;
  const result = decide(run, resp);
  // onBreach "hold": wait here for someone to raise the limit. Only a breach the server decided: an unreachable one can't be raised
  if (result.breachType && control.HOLDABLE.has(result.breachType) && control.holds(run) && resp && !resp.network_error) {
    control.startHold(run, "breach");
    await control.waitIfPaused(run);
    const outcome = control.endHold(run);
    if (outcome === "released") {  // an operator raised the limit and resumed the run
      run.lastBreach = null;
      return okResult(run.costUsd);
    }
    return outcome === "stopped" ? finalResult(run) : result;  // timed out: the limit it was held at
  }
  if (run.control.paused && !result.breached) {  // paused from the dashboard: wait here until resumed
    await control.waitIfPaused(run);
    return finalResult(run);
  }
  return result;
}

/** @internal A cost or count must be a finite number, zero or more: a NaN or negative one would switch the budget off. */
export function checkAmount(value: unknown, what: string): number {
  if (typeof value !== "number" || Number.isNaN(value)) throw new TypeError(`${what} must be a number, not ${String(value)}`);
  if (value < 0 || value === Infinity) throw new RangeError(`${what} must be zero or more and finite, not ${value}`);
  return value;
}

/** @internal The breach the run is in by local rules, if any. */
export function localBreach(run: Run): string | null {
  // Limits trip only once exceeded, same rule as the server; round away float drift (0.02*5 != 0.10)
  if (run.budgetUsd !== null && Math.round(run.costUsd * 1e9) / 1e9 > run.budgetUsd) return "budget_exceeded";
  if (run.iterations > run.maxIterations) return "iteration_limit";
  if (run.durationMs > run.maxDurationMs) return "time_limit";
  if (run.maxTokens > 0 && run.tokensUsed > run.maxTokens) return "token_limit";
  return null;
}

function decide(run: Run, resp: Reply | null): CheckpointResult {
  if (!resp || resp.network_error) {
    if (resp) {
      const ending = failClosed || !!resp.auth_rejected;  // a key revoked mid-run: the run can't go on unchecked
      warnUnreachable(run, resp.network_error, ending);
      if (ending) {
        run.lastBreach = "fences_unreachable";
        return finalResult(run);
      }
    }
    run.lastBreach = localBreach(run);
  } else {
    control.apply(run, resp.commands);
    // The server is authoritative: adopt its totals and (on a breach) its current limits
    control.adopt(run, resp, control.TOTALS);
    control.adopt(run, resp, control.LIMITS, !run.budgetInCode);
    // A fresh breach, or a 409 because the run is still fenced from an earlier one
    const breach = typeof resp.breach === "string" && resp.breach ? resp.breach : null;  // an odd one is still a breach
    run.lastBreach = resp.ok === true ? null : (breach ?? run.lastBreach ?? "limit_reached");
  }
  return finalResult(run);
}

function finalResult(run: Run): CheckpointResult {
  if (run.control.stopped) {  // a stop from the dashboard sticks, whatever a later reply says
    run.lastBreach = "stopped_by_user";
    run.control.stopSeen = true;
  }
  else if (run.control.pauseExpired) run.lastBreach = "paused";
  return run.lastBreach ? breachResult(run.lastBreach, run) : okResult(run.costUsd);
}

// ── Decision trail and spans ──────────────────────────────────────────────────

/**
 * Wait up to timeoutMs for queued decisions and spans to reach the server, and for onStop and
 * onQuarantine hooks to finish. Call before a serverless handler returns.
 */
export async function flush(timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  await outbox.wait(timeoutMs);
  await control.drain(Math.max(0, deadline - Date.now()));
}

/** Record the agent's reasoning at this step. Does nothing outside a governed run. */
export function logDecision(reasoning: string, action: string | null = null): void {
  const run = getActiveRun();
  if (!run) return;
  run.decisions.push({ timestamp: Date.now() / 1000, iteration: run.iterations, reasoning, action });
  if (run.decisions.length > MAX_EVENTS_PER_RUN) run.decisions.shift();
  if (!client) return;
  const sent = events.redact({ type: "decision", reasoning, action });
  if (sent && typeof sent.reasoning === "string" && sent.reasoning) {
    outbox.enqueue({ kind: "decision", client, runId: run.runId,
              body: { iteration: run.iterations, reasoning: sent.reasoning, action: sent.action ?? null } });
  }
}

/**
 * Add a tool or model call to the active run as an OpenTelemetry-shaped span (GenAI
 * semantic conventions): kept on the run, and sent through the redact hook in cloud mode.
 */
export function recordSpan(name: string, elapsedMs: number, attributes: Record<string, unknown>, error?: unknown): void {
  const run = getActiveRun();
  if (!run) return;
  const end = Date.now() / 1000;
  const attrs: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(attributes)) if (v !== null && v !== undefined) attrs[k] = v;
  attrs["fences.iteration"] = run.iterations;
  if (error !== undefined) {
    attrs["error.type"] = events.errorType(error);
    attrs["fences.error.message"] = events.errorText(error);
    const status = (error as { status?: unknown })?.status;  // the OpenAI and Anthropic JS SDKs put the HTTP status here
    if (typeof status === "number") attrs["http.response.status_code"] = status;
  }
  const span: Span = { name, start_ts: end - elapsedMs / 1000, end_ts: end, status: error === undefined ? "ok" : "error", attributes: attrs };
  run.events.push(span);
  if (run.events.length > MAX_EVENTS_PER_RUN) run.events.shift();  // a runaway agent can't eat memory
  if (!client || !outbox.spansSupported) return;
  const sent = events.redact({ type: "span", ...span });
  if (sent) {
    const { type: _, ...redacted } = sent;  // the type key is for the redact hook only
    outbox.enqueue({ kind: "span", client, runId: run.runId, span: redacted as unknown as Span });
  }
}
