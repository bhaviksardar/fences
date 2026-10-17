/**
 * Budget prechecks: before a model call goes out, work out the most it could cost and refuse it
 * (FencesStop) if that could take the run past its budget. The amount is held while the call is in
 * flight, so parallel calls can't overspend together, and released at the next checkpoint, which
 * records the actual cost. Used by instrument(), which sees each request before it's sent.
 * Same rules as the Python SDK's precheck.py.
 */
import * as control from "./control.js";
import { FencesStop } from "./errors.js";
import { costOf, priceOf, readUsage } from "./pricing.js";
import { breachResult } from "./results.js";
import { getActiveRun, type Hold, type Run } from "./run.js";

const DATA_URI = /data:[\w/+.-]+;base64,[A-Za-z0-9+/=\\]+/g;
const FILE_TOKENS = 2_000;       // ponytail: a flat guess per embedded image or file; providers price them by size
const BYTES_PER_TOKEN = 2;       // ponytail: tokenizers average 3-4 bytes a token for English and code; 2 keeps the estimate high
const DEFAULT_MAX_OUT = 32_768;  // for the few models whose output cap isn't in the price table
const MAX_OUT_PARAMS = ["max_completion_tokens", "max_output_tokens", "max_tokens"];

type Params = Record<string, unknown>;

/** The output-token cap a request sets, if any. */
export function maxOut(params: Params): number | null {
  for (const k of MAX_OUT_PARAMS) if (Number.isInteger(params[k])) return params[k] as number;
  return null;
}

/**
 * The most a call could cost: its input, estimated high from the request's size, at the dearest
 * input rate, plus its output cap (or the model's) at the output rate. null for a model with no
 * known price, which can't be prechecked.
 */
export function worstCaseUsd(model: unknown, request: unknown, outCap: number | null = null, n = 1): number | null {
  const p = typeof model === "string" && model ? priceOf(model) : undefined;
  if (!p) return null;
  let text: string;
  try {
    text = JSON.stringify(request, (_k, v) => typeof v === "bigint" ? String(v) : v) ?? "";
  } catch {
    text = String(request);  // circular: its size is a guess, but the call isn't broken by the check
  }
  const files = text.match(DATA_URI)?.length ?? 0;
  // TextEncoder, not Buffer: edge runtimes have no Buffer
  const tokensIn = Math.floor(new TextEncoder().encode(text.replace(DATA_URI, "")).length / BYTES_PER_TOKEN) + files * FILE_TOKENS;
  const tokensOut = (outCap ?? p.max_out ?? DEFAULT_MAX_OUT) * Math.max(n, 1);
  const rates = p.above !== undefined && p.tier && tokensIn > p.above ? { ...p, ...p.tier } : p;
  return tokensIn * Math.max(rates.in, rates.cache_write ?? 0) + tokensOut * rates.out;
}

/** Hold worst if it fits in what's left of the budget; null if it doesn't. */
function take(run: Run, worst: number): Hold | null {
  if (Math.round((run.costUsd + run.reservedUsd + worst) * 1e9) / 1e9 > run.budgetUsd!) return null;
  const h: Hold = { amount: worst, finished: false };
  run.holds.push(h);
  run.reservedUsd += worst;
  return h;
}

function refuse(run: Run, worst: number, outCap: number | null): never {
  if (run.control.stopped) {
    run.lastBreach = "stopped_by_user";
    run.control.stopSeen = true;
    throw new FencesStop(breachResult("stopped_by_user", run));
  }
  const room = Math.max(run.budgetUsd! - run.costUsd - run.reservedUsd, 0);
  const result = breachResult("budget_exceeded", run);
  result.message = `I can't make another model call without risking going over my budget (spent $${run.costUsd.toFixed(4)} ` +
    `of $${run.budgetUsd!.toFixed(4)}; the next call could cost up to $${worst.toFixed(4)}). I've stopped here.`;
  result.systemPrompt = "The run's budget can't cover another model call. Stop now; don't call any more tools or models.";
  run.lastBreach = "budget_exceeded";  // the run ends breached even if the agent catches FencesStop and returns
  console.warn(`agentfences: ${run.agentName}: refused a model call that could cost up to $${worst.toFixed(4)} with ` +
    `$${room.toFixed(4)} of its budget left.${outCap ? "" : " Setting max_tokens on the call lowers its worst case."}`);
  throw new FencesStop(result);
}

export interface Prepared { run: Run; worst: number; outCap: number | null }

/** The call to check, or null when there's nothing to check (no run, no budget, a model with no known price). */
export function prepare(model: unknown, request: unknown, params: Params = {}): Prepared | null {
  const run = getActiveRun();
  if (!run || run.budgetUsd === null) return null;
  const outCap = maxOut(params);
  const worst = worstCaseUsd(model, request, outCap, Number.isInteger(params.n) ? (params.n as number) : 1);
  return worst === null ? null : { run, worst, outCap };
}

/**
 * Before a model call, without waiting: hold its worst case, or throw FencesStop if it doesn't fit.
 * Returns "wait" when it doesn't fit but the run waits at a limit (onBreach "hold"): then use holdAsync.
 */
export function holdNow(p: Prepared): Hold | "wait" {
  const h = take(p.run, p.worst);
  if (h) return h;
  if (control.holds(p.run)) return "wait";
  refuse(p.run, p.worst, p.outCap);
}

/** The same, waiting (onBreach "hold") for someone to raise the budget while the call doesn't fit. */
export async function holdAsync(p: Prepared): Promise<Hold> {
  for (;;) {
    const h = take(p.run, p.worst);
    if (h) return h;
    if (!control.holds(p.run)) refuse(p.run, p.worst, p.outCap);
    control.startHold(p.run, "budget");
    await control.waitIfPaused(p.run);
    if (control.endHold(p.run) !== "released") refuse(p.run, p.worst, p.outCap);
  }
}

/**
 * After the call. A failed call isn't billed, so its hold is released now. Otherwise the hold
 * shrinks to the response's actual cost (a stream without usage keeps its worst case), and the
 * next checkpoint, which records that cost, releases it.
 */
export function finish(run: Run, h: Hold, response?: unknown, failed = false): void {
  const i = run.holds.indexOf(h);  // by identity: parallel calls can hold equal amounts
  if (i < 0) return;
  if (failed) {
    run.holds.splice(i, 1);
    run.reservedUsd -= h.amount;
    return;
  }
  if (response !== undefined && readUsage(response)) {
    const actual = costOf(response)[0];
    run.reservedUsd += actual - h.amount;
    h.amount = actual;
  }
  h.finished = true;
}

/** At a checkpoint, which records the actual cost: release the holds of calls that have finished. */
export function releaseFinished(run: Run): void {
  const done = run.holds.filter((h) => h.finished);
  if (!done.length) return;
  run.holds = run.holds.filter((h) => !h.finished);
  run.reservedUsd -= done.reduce((sum, h) => sum + h.amount, 0);
}
