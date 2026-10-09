import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { FencesClient, type Reply } from "./client.js";
import { AgentQuarantined, FencesStop } from "./errors.js";
import * as events from "./events.js";
import { costOf, setCustomPrices, type CustomPrice } from "./pricing.js";

// ── Results ───────────────────────────────────────────────────────────────────

export interface CheckpointResult {
  ok: boolean;
  breached: boolean;
  /** "budget_exceeded" | "iteration_limit" | "time_limit" | "token_limit" | "stopped_by_user" | ... (see PROTOCOL.md) */
  breachType: string | null;
  /** First-person text, ready to return to the user. */
  message: string;
  /** An instruction to give the model so it wraps up. */
  systemPrompt: string;
  /** The run's total spend so far. */
  spentUsd: number;
}

const SUMMARY = "I'll summarize what I found so far.";
const WRAP_UP = "Stop your current task immediately and summarize what you have found or completed so far.";
const money = (x: number) => `$${x.toFixed(4)}`;
const count = (n: number) => n.toLocaleString("en-US");

function breachResult(type: string, run: Run): CheckpointResult {
  const spent = run.costUsd, limit = run.budgetUsd ?? 0;
  const text: Record<string, [string, string]> = {
    budget_exceeded: [`I've reached my budget limit (spent ${money(spent)} of ${money(limit)}). ${SUMMARY}`,
      `You have reached your budget limit (${money(spent)} of ${money(limit)} spent). Stop your current task immediately ` +
      "and provide a clear summary of what you have found or completed so far. Tell the user you stopped due to budget and what you accomplished."],
    iteration_limit: [`I've reached my iteration limit (${run.iterations} steps). ${SUMMARY}`,
      `You have reached your iteration limit (${run.iterations} steps). ${WRAP_UP} Tell the user you stopped due to the iteration limit.`],
    time_limit: [`I've reached my time limit (${run.durationMs}ms). ${SUMMARY}`,
      `You have reached your time limit. ${WRAP_UP} Tell the user you stopped due to the time limit.`],
    token_limit: [`I've reached my token limit (${count(run.tokensUsed)} tokens). ${SUMMARY}`,
      `You have reached your token limit (${count(run.tokensUsed)} tokens used). ${WRAP_UP} Tell the user you stopped due to the token limit.`],
    stopped_by_user: [`I was stopped from the Fences dashboard. ${SUMMARY}`,
      `A person stopped this run from the Fences dashboard. ${WRAP_UP} Tell the user you were stopped by an operator.`],
    key_daily_budget: [`My daily spending cap has been reached. ${SUMMARY}`,
      `Your daily spending cap, across all of your runs, has been reached. ${WRAP_UP} Tell the user you stopped because the daily cap was reached.`],
    key_monthly_budget: [`My monthly spending cap has been reached. ${SUMMARY}`,
      `Your monthly spending cap, across all of your runs, has been reached. ${WRAP_UP} Tell the user you stopped because the monthly cap was reached.`],
    fences_unreachable: [`I can't reach Fences to confirm I'm within my limits, so I'm stopping to be safe. ${SUMMARY}`,
      `Your governance service can't be reached and you are configured to stop when that happens. ${WRAP_UP}`],
    paused: [`I was paused from the Fences dashboard and not resumed in time. ${SUMMARY}`,
      `A person paused this run from the Fences dashboard and it was not resumed in time. Stop your current task and summarize ` +
      "what you have found or completed so far. Tell the user the run was paused by an operator."],
  };
  const [message, systemPrompt] = text[type] ??
    ["Governance limit reached.", "A governance limit has been reached. Summarize what you have done so far."];
  return { ok: false, breached: true, breachType: type, message, systemPrompt, spentUsd: run.costUsd };
}

const okResult = (spentUsd = 0): CheckpointResult =>
  ({ ok: true, breached: false, breachType: null, message: "", systemPrompt: "", spentUsd });

// ── Runs ──────────────────────────────────────────────────────────────────────

/** Every limit is optional; see governed(). */
export interface Limits {
  /** Maximum spend for the run in USD. None anywhere: unlimited spend, with a warning. */
  budgetUsd?: number | null;
  /** Maximum checkpoints (steps). Default 100. */
  maxIterations?: number | null;
  /** Maximum wall-clock time in milliseconds. Default 300000 (5 minutes). */
  maxDurationMs?: number | null;
  /** Maximum tokens (input + output). 0 or unset: no limit. */
  maxTokens?: number | null;
}

const DEFAULTS = { maxIterations: 100, maxDurationMs: 300_000, maxTokens: 0 };
export const MAX_EVENTS_PER_RUN = 1000;

export interface Span {
  name: string;
  start_ts: number;
  end_ts: number;
  status: "ok" | "error";
  attributes: Record<string, unknown>;
}

export class Run {
  readonly runId = randomUUID();
  costUsd = 0;
  iterations = 0;
  tokensUsed = 0;
  readonly startedAt = Date.now();
  readonly decisions: { timestamp: number; iteration: number; reasoning: string; action: string | null }[] = [];
  /** Tool and model calls as OpenTelemetry-shaped spans, newest 1,000 kept. */
  readonly events: Span[] = [];
  lastBreach: string | null = null;
  exception?: events.ExceptionInfo;
  warned = false;

  constructor(readonly agentName: string, public budgetUsd: number | null, public maxIterations: number,
              public maxDurationMs: number, public maxTokens: number, readonly context: Record<string, string>) {}

  get durationMs(): number { return Date.now() - this.startedAt; }
}

const store = new AsyncLocalStorage<Run>();

/** The run in progress in this async context, if any. */
export function getActiveRun(): Run | null {
  return store.getStore() ?? null;
}

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
}

let client: FencesClient | null = null;
let localOnly = false;
let failClosed = false;

/** Call once at startup, before any governed function runs. */
export function init(options: InitOptions = {}): void {
  setCustomPrices(options.prices);
  events.configure(options.environment, options.release, options.redact);
  localOnly = !!options.localOnly;
  failClosed = !!options.failClosed;
  if (localOnly) {
    client = null;
    return;
  }
  if (!options.apiKey) {
    throw new Error("apiKey is required unless localOnly is set. Use init({ localOnly: true }) for local usage.");
  }
  client = new FencesClient(options.apiKey, options.endpoint ?? "http://localhost:8000");
}

/** @internal For tests: point the SDK at a fake server client. */
export function _setClient(c: FencesClient | null): void { client = c; }

function requireInit(): void {
  if (!localOnly && !client) {
    throw new Error("Call agentfences.init() before using Fences. For local usage: init({ localOnly: true })");
  }
}

function warnUnreachable(run: Run, err: string): void {
  if (run.warned) return;
  run.warned = true;
  console.warn(`agentfences: Fences backend unreachable for run ${run.runId} (${err}); enforcing limits locally`);
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
  options: Limits & { name?: string } = {},
): (...args: A) => Promise<Awaited<R>> {
  const name = options.name ?? (fn.name || "agent");
  return async (...args: A): Promise<Awaited<R>> => {
    requireInit();
    const run = await startRun(name, options);
    let error: unknown, failed = false;
    try {
      return await (store.run(run, () => fn(...args)) as Promise<Awaited<R>>);
    } catch (e) {
      [error, failed] = [e, true];
      throw e;
    } finally {
      await endRun(run, ...outcome(run, failed, error));
    }
  };
}

function outcome(run: Run, failed: boolean, error: unknown): [string, string | null] {
  if (!failed) return [run.lastBreach ? "breached" : "success", null];
  if (error instanceof FencesStop) return ["breached", null];  // a framework integration halted the agent
  run.exception = events.exceptionInfo(error);
  const e = run.exception;
  return ["error", e.message ? `${e.type}: ${e.message}` : e.type];
}

const LIMIT_KEYS = { budget_usd: "budgetUsd", max_iterations: "maxIterations", max_duration_ms: "maxDurationMs", max_tokens: "maxTokens" } as const;

async function startRun(agentName: string, limits: Limits): Promise<Run> {
  const run = new Run(agentName, limits.budgetUsd ?? null, limits.maxIterations ?? DEFAULTS.maxIterations,
    limits.maxDurationMs ?? DEFAULTS.maxDurationMs, limits.maxTokens ?? DEFAULTS.maxTokens, events.currentContext());
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
  for (const [wire, field] of Object.entries(LIMIT_KEYS)) {
    const v = resp.limits?.[wire];
    if (typeof v === "number" || (wire === "budget_usd" && v === null)) (run as any)[field] = v;  // eslint-disable-line @typescript-eslint/no-explicit-any
  }
  for (const n of resp.notices ?? []) {
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
  if (!client) return;
  // The run must end even if redaction drops its error details
  const sent = error ? events.redact({ type: "run_end", status, error, exception: run.exception }) : null;
  const body: Record<string, unknown> = { status, error: sent?.error ?? null };
  if (sent?.exception) body.exception = sent.exception;
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
  const stepCost = cost + (extra.costDeltaUsd ?? 0), stepTokens = tokens + (extra.tokensUsed ?? 0);
  run.costUsd += stepCost;
  run.tokensUsed += stepTokens;
  run.iterations += 1;
  const resp = client ? await client.checkpoint(run.runId, {
    cost_delta_usd: stepCost, tokens_used: stepTokens, iterations: run.iterations, duration_ms: run.durationMs }) : null;
  return decide(run, resp);
}

function localBreach(run: Run): string | null {
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
      warnUnreachable(run, resp.network_error);
      if (failClosed) {
        run.lastBreach = "fences_unreachable";
        return breachResult(run.lastBreach, run);
      }
    }
    run.lastBreach = localBreach(run);
  } else {
    // The server is authoritative: adopt its totals and (on a breach) its current limits
    run.costUsd = resp.spent_usd ?? run.costUsd;
    run.iterations = resp.iterations ?? run.iterations;
    run.tokensUsed = resp.tokens_used ?? run.tokensUsed;
    for (const [wire, field] of Object.entries(LIMIT_KEYS)) {
      if (wire in resp) (run as any)[field] = resp[wire];  // eslint-disable-line @typescript-eslint/no-explicit-any
    }
    // A fresh breach, or a 409 because the run is still fenced from an earlier one
    run.lastBreach = resp.ok ? null : (resp.breach ?? run.lastBreach ?? "limit_reached");
  }
  return run.lastBreach ? breachResult(run.lastBreach, run) : okResult(run.costUsd);
}

// ── Decision trail and spans ──────────────────────────────────────────────────

const SPAN_BATCH = 50;
type Item = { kind: "decision"; runId: string; body: Record<string, unknown> } | { kind: "span"; runId: string; span: Span };
const outbox: Item[] = [];
let draining: Promise<void> | null = null;
let spansSupported = true;

function enqueue(item: Item): void {
  outbox.push(item);
  draining ??= drain().finally(() => { draining = null; });
}

/** Decisions and spans go to the server in the background, in order; consecutive spans of a run go together. */
async function drain(): Promise<void> {
  while (outbox.length) {
    const item = outbox.shift()!;
    const c = client;
    if (!c) continue;
    try {
      if (item.kind === "decision") {
        await c.logDecision(item.runId, item.body);
        continue;
      }
      const batch = [item.span];
      while (outbox[0]?.kind === "span" && outbox[0].runId === item.runId && batch.length < SPAN_BATCH) {
        batch.push((outbox.shift() as Extract<Item, { kind: "span" }>).span);
      }
      if (spansSupported && (await c.logSpans(item.runId, batch)).unsupported) {
        spansSupported = false;
        console.warn("agentfences: this Fences server doesn't accept tool and model call spans yet; " +
          "they're kept on the run locally but not sent");
      }
    } catch { /* never let delivery break the agent */ }
  }
}

/** Wait up to timeoutMs for queued decisions and spans to reach the server. Call before a serverless handler returns. */
export async function flush(timeoutMs = 5000): Promise<void> {
  if (!draining) return;
  await Promise.race([draining, new Promise((r) => setTimeout(r, timeoutMs).unref())]);
}

/** Record the agent's reasoning at this step. Does nothing outside a governed run. */
export function logDecision(reasoning: string, action: string | null = null): void {
  const run = getActiveRun();
  if (!run) return;
  run.decisions.push({ timestamp: Date.now() / 1000, iteration: run.iterations, reasoning, action });
  if (!client) return;
  const sent = events.redact({ type: "decision", reasoning, action });
  if (sent && typeof sent.reasoning === "string" && sent.reasoning) {
    enqueue({ kind: "decision", runId: run.runId,
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
  if (!client || !spansSupported) return;
  const sent = events.redact({ type: "span", ...span });
  if (sent) {
    const { type: _, ...redacted } = sent;  // the type key is for the redact hook only
    enqueue({ kind: "span", runId: run.runId, span: redacted as unknown as Span });
  }
}
