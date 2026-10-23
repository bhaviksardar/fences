/** Approval requests: ask a person, through the Fences dashboard, and wait for the answer. */
import { randomUUID } from "node:crypto";
import * as control from "./control.js";
import { recordSpan } from "./core.js";
import { getActiveRun } from "./run.js";

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
  const run = getActiveRun(), client = control._client();
  const timeoutS = options.timeoutS ?? 600, amountUsd = options.amountUsd ?? null;
  if (!run) return { granted: false, note: "requestApproval() was called outside a governed run" };
  if (!client) return { granted: false, note: "Local mode: there is no one to approve this request" };
  if (!control.supported.approvals) return { granted: false, note: "This Fences server doesn't support approvals" };

  const approvalId = randomUUID(), started = performance.now();
  // timeout_s lets the server expire the request when we stop waiting
  const resp = await client.requestApproval(run.runId, { approval_id: approvalId, reason, amount_usd: amountUsd, timeout_s: timeoutS });
  if (resp.unsupported) {
    control.unsupported("approvals", "approval requests", "they are denied");
    return { granted: false, note: "This Fences server doesn't support approvals" };
  }
  if (resp.network_error) return { granted: false, note: "Couldn't reach Fences to ask for approval" };
  if (resp.ok === false) return { granted: false, note: `Fences didn't accept the approval request (HTTP ${resp.status})` };  // no one will be asked

  run.control.approvals.set(approvalId, null);
  run.control.awaiting += 1;
  control.schedule(0);  // poll fast while someone decides
  const deadline = Date.now() + timeoutS * 1000;
  while (run.control.approvals.get(approvalId) === null && Date.now() < deadline) {
    await new Promise((ok) => setTimeout(ok, 100));
  }
  run.control.awaiting -= 1;
  const c = run.control.approvals.get(approvalId);  // the dashboard's approval command, or null: no answer in time
  run.control.approvals.delete(approvalId);
  const answer: Approval = c
    // granted only when it's true, and an amount only when it's a real one
    ? { granted: c.granted === true, by: c.by ?? null, note: c.note ?? null,
        amountUsd: typeof c.amount_usd === "number" && Number.isFinite(c.amount_usd) && c.amount_usd > 0 ? c.amount_usd : null, approvalId }
    : { granted: false, note: "No answer before the timeout", approvalId };
  if (answer.granted && answer.amountUsd && run.budgetUsd !== null) run.budgetUsd += answer.amountUsd;  // an unlimited run stays unlimited
  recordSpan("fences.approval", performance.now() - started, {  // one span from request to answer
    "fences.approval.id": approvalId, "fences.approval.reason": reason, "fences.approval.amount_usd": amountUsd,
    "fences.approval.granted": answer.granted, "fences.approval.by": answer.by, "fences.approval.note": answer.note,
    "fences.approval.granted_usd": answer.amountUsd,
  });
  return answer;
}
