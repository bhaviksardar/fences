import { randomUUID } from "node:crypto";
import * as approvals from "./approvals.js";
import * as control from "./control.js";
import { _client, checkAmount, getActiveRun, localBreach, type Run } from "./core.js";
import { ToolBlocked } from "./errors.js";
import * as events from "./events.js";
import { MAX_ARGS, recordToolCall } from "./spans.js";

/** `query, limit` (one object argument: its keys), or `0, 1` (positions): what a tool call span carries by default. */
export function argNames(args: unknown[]): string {
  const [only] = args;
  if (args.length === 1 && only && typeof only === "object" && !Array.isArray(only)) return Object.keys(only).join(", ");
  return args.map((_, i) => String(i)).join(", ");
}

/** `{"query":"cats","limit":5}`: the arguments as compact JSON (one object argument on its own), capped. */
export function argsSummary(args: unknown[]): string {
  let text: string;
  try {
    text = JSON.stringify(args.length === 1 ? args[0] : args, (_k, v) => typeof v === "bigint" ? String(v) : v) ?? "";
  } catch {
    text = args.map(String).join(", ");  // circular or odd values
  }
  return text.length <= MAX_ARGS ? text : text.slice(0, MAX_ARGS - 1) + "…";
}

// ── Guards ────────────────────────────────────────────────────────────────────

function block(run: Run, name: string, reason: string, breach: string | null = null): never {
  if (run.control.stopped) run.control.stopSeen = true;  // the agent learns of the stop here, so don't cancel it
  throw new ToolBlocked(name, reason, breach);
}

const fits = (run: Run, amount: number) =>
  run.budgetUsd === null || Math.round((run.costUsd + run.reservedUsd + amount) * 1e9) / 1e9 <= run.budgetUsd;

async function approve(run: Run, name: string, summary: string, amount: number): Promise<string> {
  const answer = await approvals.requestApproval(`${name}(${summary})`, { amountUsd: amount || undefined });
  if (!answer.granted) block(run, name, answer.note ? `not approved: ${answer.note}` : "not approved");
  return answer.approvalId!;
}

/** Check a guarded call and reserve its amount; returns the guard id, or throws ToolBlocked. */
async function open(run: Run, name: string, summary: string, amount: number, needsApproval: boolean): Promise<string> {
  checkAmount(amount, `${name}'s costUsd`);  // a NaN or negative amount would let calls past the budget
  if (run.control.quarantined) block(run, name, "the agent is quarantined in Fences");  // now, not when the quarantine's pause ends
  if (run.control.paused) await control.waitIfPaused(run);
  if (run.control.pauseExpired) block(run, name, "the run was paused from the Fences dashboard and not resumed in time", "paused");
  const approvalId = needsApproval ? await approve(run, name, summary, amount) : null;
  if (run.control.stopped) block(run, name, "the run was stopped from the Fences dashboard", "stopped_by_user");
  const client = _client();
  if (!client) {  // local mode: the same rules, applied here
    const breach = run.lastBreach ?? localBreach(run);
    if (breach) block(run, name, `the run is over a limit (${breach})`, breach);
    if (!fits(run, amount)) block(run, name, `$${amount.toFixed(2)} more would take the run past its $${run.budgetUsd!.toFixed(2)} budget`, "budget_exceeded");
    run.reservedUsd += amount;
    return randomUUID();
  }
  const call = { guardId: randomUUID(), approvalId };  // one id per call: the server answers a re-check of a denied id afresh
  for (;;) {
    try {
      return await check(run, client, name, summary, amount, call);
    } catch (e) {
      if (!(e instanceof ToolBlocked && control.BUDGET_BREACHES.has(e.breachType ?? "") && control.holds(run))) throw e;
      control.startHold(run, "budget");  // onBreach "hold": wait for someone to raise the budget, then check again
      await control.waitIfPaused(run);
      const outcome = control.endHold(run);
      if (outcome === "stopped") block(run, name, "the run was stopped from the Fences dashboard", "stopped_by_user");
      if (outcome !== "released") throw e;
    }
  }
}

async function check(run: Run, client: NonNullable<ReturnType<typeof _client>>, name: string, summary: string, amount: number,
                     call: { guardId: string; approvalId: string | null }): Promise<string> {
  if (run.control.stopped) block(run, name, "the run was stopped from the Fences dashboard", "stopped_by_user");
  // Check against what this process already holds before asking the server, which can't see it; reserve until it answers
  if (!fits(run, amount)) block(run, name, `$${amount.toFixed(2)} more could take the run past its $${run.budgetUsd!.toFixed(2)} budget`, "budget_exceeded");
  run.reservedUsd += amount;
  try {
    const sent: Record<string, unknown> = events.redact({ type: "guard", tool: name, arguments: summary }) ?? {};  // the check goes ahead without them
    let resp: Record<string, any> = {};  // eslint-disable-line @typescript-eslint/no-explicit-any
    for (let attempt = 0; attempt < 2; attempt++) {  // once more after an approval the server asked for
      resp = await client.guard(run.runId, { guard_id: call.guardId, phase: "check", tool: name, arguments: sent.arguments ?? null,
                                             amount_usd: amount, approval_id: call.approvalId });
      control.apply(run, resp.commands);
      if (resp.allow === true) return call.guardId;
      if (!(resp.needs_approval && call.approvalId === null)) break;
      call.approvalId = await approve(run, name, summary, amount);
    }
    if (resp.unsupported) {
      control.unsupported("guard", "guarded tools", "they are blocked");
      block(run, name, "this Fences server doesn't support guarded tools");
    }
    if (resp.network_error || !("allow" in resp)) {  // no answer: fail closed
      block(run, name, resp.status === 401 || resp.status === 403 ? "Fences rejected the API key" : "couldn't reach Fences to check this call");
    }
    block(run, name, resp.reason ?? "not allowed", resp.breach ?? null);
  } catch (e) {
    run.reservedUsd -= amount;
    throw e;
  }
}

/** Release the reservation and charge the run: in full unless the tool threw (a timeout may have done it). */
async function settle(run: Run, guardId: string, amount: number, error: unknown, failed: boolean): Promise<void> {
  const outcome = !failed ? "ok" : (error as Error)?.name === "TimeoutError" ? "unknown" : "error";
  const charged = outcome === "error" ? 0 : amount;
  run.reservedUsd -= amount;
  run.costUsd += charged;
  await _client()?.guard(run.runId, { guard_id: guardId, phase: "settle", outcome, cost_usd: charged });
}

/** Record an async tool call; cost is null for an unguarded call, else a function giving its amount. */
/** summary (the values) goes to the guard and an approval request; the span gets names. */
export async function callAsync<R>(run: Run, name: string, summary: string, cost: (() => number) | null, needsApproval: boolean,
                                   call: () => Promise<R>, names?: string): Promise<R> {
  const started = performance.now();
  let gate: [string, number] | null = null;
  try {
    if (cost) {
      const amount = cost();
      gate = [await open(run, name, summary, amount, needsApproval), amount];
    }
    const out = await call();
    recordToolCall(name, summary, started, undefined, { argNames: names });
    if (gate) await settle(run, ...gate, undefined, false);
    return out;
  } catch (e) {
    recordToolCall(name, summary, started, e, { argNames: names });
    if (gate) await settle(run, ...gate, e, true);
    throw e;
  }
}

// ── tool() ────────────────────────────────────────────────────────────────────

interface ToolFields<A extends unknown[]> {
  name?: string;
  /** Reject with a TimeoutError if the call takes longer. JS can't stop the tool's code; pass run.signal to its fetches. */
  timeoutS?: number;
  /** Ask Fences before each call; it runs only if allowed. */
  guard?: boolean;
  /** What a call costs the run, reserved before and charged after: a number, or a function of the tool's arguments. */
  costUsd?: number | ((...args: A) => number);
  /** A person approves each call from the dashboard first. */
  approve?: boolean;
}
/** Options that make a tool async: a guard (guard, costUsd or approve) or a timeout. */
export type GuardedToolOptions<A extends unknown[]> = ToolFields<A> &
  ({ guard: true } | { costUsd: number | ((...args: A) => number) } | { approve: true } | { timeoutS: number });

function timed<R>(p: Promise<R>, timeoutS: number | undefined, name: string): Promise<R> {
  if (timeoutS === undefined) return p;
  let timer: NodeJS.Timeout;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new DOMException(`${name} took longer than ${timeoutS}s`, "TimeoutError")), timeoutS * 1000);
  });
  return Promise.race([p, late]).finally(() => clearTimeout(timer));
}

/**
 * Record every call of an agent tool on the active run:
 *
 *     const webSearch = tool(async (query: string) => { ... });
 *     const fetchPage = tool(fetchPageImpl, { name: "fetch", timeoutS: 30 });
 *
 * Records the tool's name, a short summary of its arguments, whether it succeeded (and the
 * error if not) and how long it took. Errors still reach the caller; sync tools stay sync.
 * Outside a governed run it just calls the function.
 *
 * Guarded tools ask Fences before they run, for actions with consequences:
 *
 *     const issueRefund = tool(refund, { costUsd: (orderId, amount) => amount, approve: true });
 *
 * guard: true runs the tool only if the run isn't stopped, paused, over a limit or
 * quarantined; otherwise it throws ToolBlocked and the tool's code never runs. It's blocked
 * too if Fences can't be reached (guards fail closed). costUsd is reserved before the call,
 * so parallel calls can't overspend together, and charged after (unless the tool threw; a
 * timeout is charged, as it may have happened). approve: true asks a person first.
 * Setting costUsd or approve guards the tool. A guarded or timed tool returns a promise.
 */
export function tool<A extends unknown[], R>(fn: (...args: A) => R, options: GuardedToolOptions<A>): (...args: A) => Promise<Awaited<R>>;
export function tool<A extends unknown[], R>(fn: (...args: A) => R, options?: { name?: string }): (...args: A) => R;
export function tool<A extends unknown[], R>(fn: (...args: A) => R, options: ToolFields<A> = {}): (...args: A) => R | Promise<Awaited<R>> {
  const name = options.name ?? (fn.name || "tool");
  const guarded = !!options.guard || options.costUsd !== undefined || !!options.approve;
  if (guarded || options.timeoutS !== undefined) {
    return async (...args: A): Promise<Awaited<R>> => {
      const run = getActiveRun();
      if (!run) return timed(Promise.resolve().then(() => fn(...args)), options.timeoutS, name) as Promise<Awaited<R>>;
      const c = options.costUsd;
      return callAsync(run, name, argsSummary(args), guarded ? () => (typeof c === "function" ? c(...args) : c) ?? 0 : null,
                       !!options.approve, () => timed(Promise.resolve().then(() => fn(...args)), options.timeoutS, name) as Promise<Awaited<R>>,
                       argNames(args));
    };
  }
  return (...args: A): R => {
    if (!getActiveRun()) return fn(...args);
    const summary = argsSummary(args), started = performance.now(), opts = { argNames: argNames(args) };
    let out: R;
    try {
      out = fn(...args);
    } catch (e) {
      recordToolCall(name, summary, started, e, opts);
      throw e;
    }
    if (out instanceof Promise) {
      return out.then((v) => { recordToolCall(name, summary, started, undefined, opts); return v; },
                      (e) => { recordToolCall(name, summary, started, e, opts); throw e; }) as R;
    }
    recordToolCall(name, summary, started, undefined, opts);
    return out;
  };
}
