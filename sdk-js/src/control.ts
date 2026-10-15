/**
 * Live control from the Fences dashboard: a heartbeat that reports live runs and brings back
 * commands (stop, pause, resume, quarantine, limits, approval answers) and acknowledges them,
 * containment hooks, pausing at checkpoints, cancelling a run stuck after a stop, and approval
 * requests. Cloud mode only; a server without these endpoints gets one warning and runs carry
 * on as before. Same behaviour as the Python SDK (PROTOCOL.md, "Live control").
 */
import { randomUUID } from "node:crypto";
import type { FencesClient, Reply } from "./client.js";
import type { Run } from "./core.js";

/** Heartbeat statuses whose acks go again with the next heartbeat: rate limited, or a server error. */
const retryable = (status: number | undefined) => status === undefined || status === 429 || status >= 500;
const FAST_MS = 2000;  // heartbeat interval while a run is paused, waiting for an approval, or stopping
const HOOK_TIMEOUT_MS = 30_000;  // a hook still running after this is reported as timed out

export interface Approval {
  granted: boolean;
  /** Who answered, as the dashboard reports it. */
  by?: string | null;
  /** Their note, or why no one could answer. */
  note?: string | null;
  /** The amount approved, if one was asked for. */
  amountUsd?: number | null;
  approvalId?: string | null;
}

/** What onStop and onQuarantine receive. */
export interface ControlEvent {
  type: "stop" | "quarantine";
  runId: string;
  agentName: string;
  /** The command's id, if the server sent one. */
  id: string | null;
  /** Who pressed the button, as the dashboard reports it. */
  by: string | null;
  reason: string | null;
}
export type Hook = (event: ControlEvent) => unknown;

/** Live-control state a run carries. */
export interface Control {
  stopped: boolean;
  stoppedAt: number;
  /** The agent got its stop from a checkpoint (or a blocked tool): it's never cancelled. */
  stopSeen: boolean;
  cancelled: boolean;
  /** Set by governed(): halts the call when a stop goes unseen past stopGraceS. */
  onCancel: (() => void) | null;
  paused: boolean;
  pauseExpired: boolean;
  quarantined: boolean;
  /** While waiting at a limit: "breach" or "budget". */
  held: "breach" | "budget" | null;
  /** approval id -> the answer, or null while waiting */
  approvals: Map<string, Approval | null>;
  awaiting: number;
  /** command ids already acknowledged */
  acked: Set<string>;
}

export const newControl = (): Control => ({
  stopped: false, stoppedAt: 0, stopSeen: false, cancelled: false, onCancel: null, paused: false, pauseExpired: false,
  quarantined: false, held: null, approvals: new Map(), awaiting: 0, acked: new Set(),
});

const live = new Map<string, Run>();  // run id -> runs in progress
let client: FencesClient | null = null;
let intervalMs = 15_000;
export let pauseTimeoutMs = 3_600_000;
let stopGraceMs: number | null = 30_000;
const hooks: { stop: Hook | null; quarantine: Hook | null } = { stop: null, quarantine: null };
export const supported = { heartbeat: true, approvals: true, guard: true };
let timer: NodeJS.Timeout | null = null;
let beating = false;
let acks: Record<string, unknown>[] = [];   // acknowledgements waiting for the next heartbeat
const hooked = new Set<string>();           // quarantine ids whose hook has run in this process
const runningHooks = new Set<Promise<void>>();

export interface ControlOptions {
  heartbeatS: number;
  pauseTimeoutS: number;
  stopGraceS: number | null;
  onStop?: Hook | null;
  onQuarantine?: Hook | null;
}

export function configure(c: FencesClient | null, o: ControlOptions): void {
  client = c;
  intervalMs = o.heartbeatS * 1000;
  pauseTimeoutMs = o.pauseTimeoutS * 1000;
  stopGraceMs = o.stopGraceS === null ? null : o.stopGraceS * 1000;
  hooks.stop = o.onStop ?? null;
  hooks.quarantine = o.onQuarantine ?? null;
  supported.heartbeat = supported.approvals = supported.guard = true;
  schedule(0);  // a new client or interval takes effect now, not after the old wait
}

export function register(run: Run): void {
  live.set(run.runId, run);
  if (!timer) schedule(nextDelay());
}

export function unregister(run: Run): void {
  live.delete(run.runId);
}

export function unsupported(kind: keyof typeof supported, what: string, consequence: string): void {
  if (!supported[kind]) return;
  supported[kind] = false;
  console.warn(`agentfences: this Fences server doesn't support ${what} yet, so ${consequence}`);
}

// ── Commands ──────────────────────────────────────────────────────────────────

const LIMITS = { budget_usd: "budgetUsd", max_iterations: "maxIterations", max_duration_ms: "maxDurationMs", max_tokens: "maxTokens" } as const;

/** Apply the commands addressed to this run (from a heartbeat or a checkpoint reply), and acknowledge them. Each is safe to apply twice. */
export function apply(run: Run, commands: Reply[] | undefined): void {
  // Limits first: a resume in the same reply may release a call held for budget, which re-checks against them at once
  for (const c of [...(commands ?? [])].sort((a, b) => Number(a.type !== "limits") - Number(b.type !== "limits"))) {
    if (c.run_id !== undefined && c.run_id !== run.runId) continue;
    const ctl = run.control;
    if (c.type === "stop") {
      if (!ctl.stopped) {
        ctl.stopped = true;
        ctl.stoppedAt = Date.now();
        schedule(0);  // watch for the agent to see it, or cancel it
        hook("stop", run, c);  // once per run
      }
      ctl.paused = ctl.quarantined = false;
    } else if (c.type === "pause" || c.type === "quarantine") {  // a quarantined agent's runs hold at their next checkpoint
      ctl.paused = true;
      schedule(0);
      if (c.type === "quarantine") {
        ctl.quarantined = true;  // guarded tools are blocked here too, even if the server can't be reached
        hook("quarantine", run, c);
      }
    } else if (c.type === "resume") {
      ctl.paused = ctl.quarantined = false;  // releasing a quarantine resumes its runs
    } else if (c.type === "limits") {
      for (const [wire, field] of Object.entries(LIMITS)) {
        if (c[wire] !== undefined && c[wire] !== null) (run as any)[field] = c[wire];  // eslint-disable-line @typescript-eslint/no-explicit-any
      }
    } else if (c.type === "approval") {
      if (ctl.approvals.has(c.approval_id)) {
        ctl.approvals.set(c.approval_id, { granted: !!c.granted, by: c.by ?? null, note: c.note ?? null, amountUsd: c.amount_usd ?? null });
      }
    } else {
      continue;  // unknown: ignored, and not acknowledged
    }
    if (c.id && !ctl.acked.has(c.id)) {
      ctl.acked.add(c.id);
      ack({ run_id: run.runId, id: c.id, type: c.type, received_at: Date.now() / 1000 });
    }
  }
}

// ── Acknowledgements and containment hooks ────────────────────────────────────

function ack(entry: Record<string, unknown>): void {
  acks.push(entry);
  schedule(0);  // send it now, not at the next interval
}

function hook(kind: "stop" | "quarantine", run: Run, c: Reply): void {
  const fn = hooks[kind];
  if (!fn) return;
  if (kind === "quarantine") {
    const key = c.id ?? `run:${run.runId}`;  // every server that sends quarantine gives it an id
    if (hooked.has(key)) return;  // once per process per quarantine
    hooked.add(key);
  }
  const event: ControlEvent = { type: kind, runId: run.runId, agentName: run.agentName, id: c.id ?? null, by: c.by ?? null, reason: c.reason ?? null };
  const p = runHook(fn, event).finally(() => runningHooks.delete(p));
  runningHooks.add(p);
}

async function runHook(fn: Hook, event: ControlEvent): Promise<void> {
  const { errorText } = await import("./events.js");
  let result = "ok", detail: string | null = null;
  try {
    const timedOut = Symbol("timeout");
    const out = await Promise.race([
      Promise.resolve().then(() => fn(event)),
      new Promise((ok) => setTimeout(ok, HOOK_TIMEOUT_MS, timedOut).unref()),  // ponytail: a hung hook can't be stopped; it's reported and left running
    ]);
    if (out === timedOut) result = "timeout";
  } catch (e) {
    [result, detail] = ["failed", errorText(e).slice(0, 300)];
  }
  if (result !== "ok") console.warn(`agentfences: the on${event.type === "stop" ? "Stop" : "Quarantine"} hook ${result === "timeout" ? "timed out" : `failed: ${detail}`}`);
  if (event.id) {
    ack({ run_id: event.runId, id: event.id, type: event.type, hook: result, ...(detail ? { detail } : {}), finished_at: Date.now() / 1000 });
  }
}

/**
 * Wait for running hooks, then send the acknowledgements still waiting. flush() calls it, and
 * it runs before the process exits, so a script that ends right after a stop still finishes
 * its containment, and says so.
 */
export async function drain(timeoutMs = HOOK_TIMEOUT_MS): Promise<void> {
  if (runningHooks.size) {
    await Promise.race([Promise.allSettled([...runningHooks]), new Promise((ok) => setTimeout(ok, timeoutMs).unref())]);
  }
  await sendAcks();
}

async function sendAcks(): Promise<void> {
  if (!acks.length || !client || !supported.heartbeat) return;
  const batch = acks;
  acks = [];
  const resp = await client.heartbeat([], batch).catch(() => ({ network_error: "heartbeat failed" }) as Reply);
  if (resp.network_error || (resp.ok === false && retryable(resp.status))) acks = [...batch, ...acks];
}

// Node exits once nothing keeps it busy: beforeExit lets acks queued by a finished hook still go out.
// One try per batch: a server that's gone mustn't keep the process alive by failing and requeueing forever.
let triedAtExit = false;
if (typeof process?.on === "function") {  // edge runtimes have no process events: flush() still drains there
  process.on("beforeExit", () => {  // fires once nothing else is pending, so every hook has finished by now
    if (triedAtExit || !acks.length || !client) return;
    triedAtExit = true;
    void sendAcks();
  });
}

// ── Heartbeat ─────────────────────────────────────────────────────────────────

const unseenStop = (r: Run) => r.control.stopped && !r.control.stopSeen && !r.control.cancelled && r.control.onCancel !== null;
const busy = () => [...live.values()].some((r) => r.control.paused || r.control.awaiting > 0 || unseenStop(r));
const nextDelay = () => (busy() ? Math.min(FAST_MS, intervalMs, stopGraceMs ?? FAST_MS) : intervalMs);  // faster while busy, never slower

function schedule(delayMs: number): void {
  if (timer) clearTimeout(timer);
  timer = null;
  if (!client || !supported.heartbeat || !(live.size || acks.length)) return;
  timer = setTimeout(beat, delayMs);
  timer.unref();  // the heartbeat never keeps the process alive on its own
}

/** Halt governed calls that haven't reached a checkpoint within stopGraceS of a stop. */
function cancelOverdue(): void {
  if (stopGraceMs === null) return;
  for (const r of live.values()) {
    if (unseenStop(r) && Date.now() - r.control.stoppedAt >= stopGraceMs) {
      r.control.cancelled = true;
      r.control.onCancel!();
    }
  }
}

async function beat(): Promise<void> {
  timer = null;
  cancelOverdue();
  if (beating || !client || !(live.size || acks.length)) return schedule(nextDelay());
  beating = true;
  const runs = [...live.values()], batch = acks;
  acks = [];
  try {
    const resp = await client.heartbeat(runs.map((r) => ({ run_id: r.runId, iterations: r.iterations, spent_usd: r.costUsd,
      tokens_used: r.tokensUsed, paused: r.control.paused, stopped: r.control.stopped,
      ...(r.control.held ? { held: r.control.held } : {}) })), batch);
    if (resp.network_error || (resp.ok === false && !resp.unsupported)) {
      if (resp.network_error || retryable(resp.status)) acks = [...batch, ...acks];  // at least once; any other 4xx won't get better
    } else if (resp.unsupported) {
      unsupported("heartbeat", "the heartbeat", "a stop from the dashboard applies at the next checkpoint and pause isn't available");
      return;
    } else {
      for (const run of runs) apply(run, (resp.commands as Reply[] | undefined)?.filter((c) => c.run_id === run.runId));
    }
  } catch {
    acks = [...batch, ...acks];  // a failed heartbeat is retried next time
  } finally {
    beating = false;
  }
  schedule(nextDelay());
}

// ── Waiting at a limit ────────────────────────────────────────────────────────

/** Breaches a person can lift by raising a limit, so onBreach "hold" waits at them. */
export const HOLDABLE = new Set(["budget_exceeded", "iteration_limit", "time_limit", "token_limit", "key_daily_budget",
                                 "key_monthly_budget", "limit_reached"]);
export const BUDGET_BREACHES = new Set(["budget_exceeded", "key_daily_budget", "key_monthly_budget"]);

/** Whether this run waits at a limit: onBreach "hold", in cloud mode (someone can raise it), and not stopped. */
export const holds = (run: Run) => run.onBreach === "hold" && client !== null && !run.control.stopped;

/** Pause the run at a limit: "breach" (a checkpoint breached) or "budget" (a call refused for budget). */
export function startHold(run: Run, kind: "breach" | "budget"): void {
  run.control.held = kind;
  if (!run.control.paused) {
    run.control.paused = true;
    schedule(0);
  }
}

/** After waiting: "released" (an operator raised the limit), "stopped", or "timeout". */
export function endHold(run: Run): "released" | "stopped" | "timeout" {
  run.control.held = null;
  if (run.control.stopped) return "stopped";
  if (run.control.pauseExpired) {
    run.control.pauseExpired = false;  // the run ends with the limit it was held at, not "paused"
    return "timeout";
  }
  return "released";
}

// ── Pausing ───────────────────────────────────────────────────────────────────

/** Hold the agent at its checkpoint while its run is paused, up to pauseTimeoutS. */
export async function waitIfPaused(run: Run): Promise<void> {
  const deadline = Date.now() + pauseTimeoutMs;
  while (run.control.paused) {
    if (Date.now() >= deadline) {
      run.control.paused = false;
      run.control.pauseExpired = true;
      return;
    }
    await new Promise((ok) => setTimeout(ok, 200));
  }
}

// ── Approvals ─────────────────────────────────────────────────────────────────

/**
 * Ask a person, through the Fences dashboard, to approve something, and wait for the answer:
 *
 *     const approval = await requestApproval("Refund $240 to order 1182?", { amountUsd: 240 });
 *     if (!approval.granted) return `I didn't issue the refund: ${approval.note}`;
 *
 * If an amount is approved, the run's budget goes up by that much. Returns denied right away in
 * local mode, outside a run, or when the server can't take requests, and denied after timeoutS
 * (default 600) with no answer: it never hangs and never throws.
 */
export async function requestApproval(reason: string, options: { amountUsd?: number; timeoutS?: number } = {}): Promise<Approval> {
  const { getActiveRun, recordSpan } = await import("./core.js");
  const run = getActiveRun();
  const timeoutS = options.timeoutS ?? 600, amountUsd = options.amountUsd ?? null;
  if (!run) return { granted: false, note: "requestApproval() was called outside a governed run" };
  if (!client) return { granted: false, note: "Local mode: there is no one to approve this request" };
  if (!supported.approvals) return { granted: false, note: "This Fences server doesn't support approvals" };

  const approvalId = randomUUID(), started = performance.now();
  // timeout_s lets the server expire the request when we stop waiting
  const resp = await client.requestApproval(run.runId, { approval_id: approvalId, reason, amount_usd: amountUsd, timeout_s: timeoutS });
  if (resp.unsupported) {
    unsupported("approvals", "approval requests", "they are denied");
    return { granted: false, note: "This Fences server doesn't support approvals" };
  }
  if (resp.network_error) return { granted: false, note: "Couldn't reach Fences to ask for approval" };

  run.control.approvals.set(approvalId, null);
  run.control.awaiting += 1;
  schedule(0);  // poll fast while someone decides
  const deadline = Date.now() + timeoutS * 1000;
  while (run.control.approvals.get(approvalId) === null && Date.now() < deadline) {
    await new Promise((ok) => setTimeout(ok, 100));
  }
  run.control.awaiting -= 1;
  const answer = run.control.approvals.get(approvalId) ?? { granted: false, note: "No answer before the timeout" };
  answer.approvalId = approvalId;
  run.control.approvals.delete(approvalId);
  if (answer.granted && answer.amountUsd && run.budgetUsd !== null) run.budgetUsd += answer.amountUsd;  // an unlimited run stays unlimited
  recordSpan("fences.approval", performance.now() - started, {  // one span from request to answer
    "fences.approval.id": approvalId, "fences.approval.reason": reason, "fences.approval.amount_usd": amountUsd,
    "fences.approval.granted": answer.granted, "fences.approval.by": answer.by, "fences.approval.note": answer.note,
    "fences.approval.granted_usd": answer.amountUsd,
  });
  return answer;
}
