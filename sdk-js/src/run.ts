/** A run's state, and which run is active in this async context. */
import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import * as control from "./control.js";
import type * as events from "./events.js";

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

export const DEFAULTS = { maxIterations: 100, maxDurationMs: 300_000, maxTokens: 0 };
export const MAX_EVENTS_PER_RUN = 1000;

export interface Span {
  name: string;
  start_ts: number;
  end_ts: number;
  status: "ok" | "error";
  attributes: Record<string, unknown>;
}

/** A model call's budget hold: its worst case, then its actual cost once it finishes. */
export interface Hold { amount: number; finished: boolean }

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
  /** Live control from the dashboard (see control.ts). */
  readonly control = control.newControl();
  /** Held by guarded tools and model calls in flight. */
  reservedUsd = 0;
  /** Model calls' worst-case costs (see precheck.ts), released at the next checkpoint once they finish. */
  holds: Hold[] = [];
  /** governed() set a budget: no server reply can make the run unlimited. */
  budgetInCode = false;
  /** "hold": wait at a limit for someone to raise it (see InitOptions.onBreach). */
  onBreach: "hold" | null = null;
  private readonly abort = new AbortController();
  /**
   * Aborts when Fences halts the run after a stop it didn't reach a checkpoint for (see
   * stopGraceS). Pass it to fetch and other cancellable work so a stuck run really stops:
   *     await fetch(url, { signal: getActiveRun()?.signal });
   */
  get signal(): AbortSignal { return this.abort.signal; }
  /** @internal */
  _abort(reason: unknown): void { this.abort.abort(reason); }

  constructor(readonly agentName: string, public budgetUsd: number | null, public maxIterations: number,
              public maxDurationMs: number, public maxTokens: number, readonly context: Record<string, string>) {}

  get durationMs(): number { return Date.now() - this.startedAt; }
}

export const store = new AsyncLocalStorage<Run>();

/** The run in progress in this async context, if any. */
export function getActiveRun(): Run | null {
  return store.getStore() ?? null;
}
