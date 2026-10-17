/** What checkpoint() returns, and the words a breach comes with: for the user, and for the model. */
import type { Run } from "./run.js";

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

export function breachResult(type: string, run: Run): CheckpointResult {
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

export const okResult = (spentUsd = 0): CheckpointResult =>
  ({ ok: true, breached: false, breachType: null, message: "", systemPrompt: "", spentUsd });
