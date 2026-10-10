/**
 * Live control from the Fences dashboard: a heartbeat that reports live runs and brings back
 * commands (stop, pause, resume, limits, approval answers), pausing at checkpoints, and
 * approval requests. Cloud mode only; a server without these endpoints gets one warning and
 * runs carry on as before. Same behaviour as the Python SDK (PROTOCOL.md, "Live control").
 */
import { randomUUID } from "node:crypto";
import type { FencesClient, Reply } from "./client.js";
import type { Run } from "./core.js";

const FAST_MS = 2000;  // heartbeat interval while a run is paused or waiting for an approval

export interface Approval {
  granted: boolean;
  /** Who answered, as the dashboard reports it. */
  by?: string | null;
  /** Their note, or why no one could answer. */
  note?: string | null;
  /** The amount approved, if one was asked for. */
  amountUsd?: number | null;
}

/** Live-control state a run carries. */
export interface Control {
  stopped: boolean;
  paused: boolean;
  pauseExpired: boolean;
  /** approval id -> the answer, or null while waiting */
  approvals: Map<string, Approval | null>;
  awaiting: number;
}

export const newControl = (): Control => ({ stopped: false, paused: false, pauseExpired: false, approvals: new Map(), awaiting: 0 });

const live = new Map<string, Run>();  // run id -> runs in progress
let client: FencesClient | null = null;
let intervalMs = 15_000;
export let pauseTimeoutMs = 3_600_000;
const supported = { heartbeat: true, approvals: true };
let timer: NodeJS.Timeout | null = null;
let beating = false;

export function configure(c: FencesClient | null, heartbeatS: number, pauseTimeoutS: number): void {
  client = c;
  intervalMs = heartbeatS * 1000;
  pauseTimeoutMs = pauseTimeoutS * 1000;
  supported.heartbeat = supported.approvals = true;
  schedule(0);  // a new client or interval takes effect now, not after the old wait
}

export function register(run: Run): void {
  live.set(run.runId, run);
  if (!timer) schedule(nextDelay());
}

export function unregister(run: Run): void {
  live.delete(run.runId);
}

function unsupported(kind: keyof typeof supported, what: string, consequence: string): void {
  if (!supported[kind]) return;
  supported[kind] = false;
  console.warn(`agentfences: this Fences server doesn't support ${what} yet, so ${consequence}`);
}

// ── Commands ──────────────────────────────────────────────────────────────────

const LIMITS = { budget_usd: "budgetUsd", max_iterations: "maxIterations", max_duration_ms: "maxDurationMs", max_tokens: "maxTokens" } as const;

/** Apply the commands addressed to this run (from a heartbeat or a checkpoint reply). Each is safe to apply twice. */
export function apply(run: Run, commands: Reply[] | undefined): void {
  for (const c of commands ?? []) {
    if (c.run_id !== undefined && c.run_id !== run.runId) continue;
    const ctl = run.control;
    if (c.type === "stop") {
      ctl.stopped = true;
      ctl.paused = false;
    } else if (c.type === "pause") {
      ctl.paused = true;
      schedule(0);
    } else if (c.type === "resume") {
      ctl.paused = false;
    } else if (c.type === "limits") {
      for (const [wire, field] of Object.entries(LIMITS)) {
        if (c[wire] !== undefined && c[wire] !== null) (run as any)[field] = c[wire];  // eslint-disable-line @typescript-eslint/no-explicit-any
      }
    } else if (c.type === "approval" && ctl.approvals.has(c.approval_id)) {
      ctl.approvals.set(c.approval_id, { granted: !!c.granted, by: c.by ?? null, note: c.note ?? null, amountUsd: c.amount_usd ?? null });
    }
  }
}

const busy = () => [...live.values()].some((r) => r.control.paused || r.control.awaiting > 0);
const nextDelay = () => (busy() ? Math.min(FAST_MS, intervalMs) : intervalMs);  // faster while busy, never slower

function schedule(delayMs: number): void {
  if (timer) clearTimeout(timer);
  timer = null;
  if (!client || !supported.heartbeat || !live.size) return;
  timer = setTimeout(beat, delayMs);
  timer.unref();  // the heartbeat never keeps the process alive on its own
}

async function beat(): Promise<void> {
  timer = null;
  if (beating || !client || !live.size) return schedule(nextDelay());
  beating = true;
  try {
    const runs = [...live.values()];
    const resp = await client.heartbeat(runs.map((r) => ({ run_id: r.runId, iterations: r.iterations, spent_usd: r.costUsd,
                                                          tokens_used: r.tokensUsed, paused: r.control.paused })));
    if (resp.unsupported) {
      unsupported("heartbeat", "the heartbeat", "a stop from the dashboard applies at the next checkpoint and pause isn't available");
      return;
    }
    for (const run of runs) apply(run, (resp.commands as Reply[] | undefined)?.filter((c) => c.run_id === run.runId));
  } catch { /* a failed heartbeat is retried next time */ } finally {
    beating = false;
  }
  schedule(nextDelay());
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
  run.control.approvals.delete(approvalId);
  if (answer.granted && answer.amountUsd && run.budgetUsd !== null) run.budgetUsd += answer.amountUsd;  // an unlimited run stays unlimited
  recordSpan("fences.approval", performance.now() - started, {  // one span from request to answer
    "fences.approval.id": approvalId, "fences.approval.reason": reason, "fences.approval.amount_usd": amountUsd,
    "fences.approval.granted": answer.granted, "fences.approval.by": answer.by, "fences.approval.note": answer.note,
    "fences.approval.granted_usd": answer.amountUsd,
  });
  return answer;
}
