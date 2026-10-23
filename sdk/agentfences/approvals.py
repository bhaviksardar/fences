"""Approval requests: ask a person, through the Fences dashboard, and wait for the answer."""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from . import control
from .core import record_span
from .run import get_active_run


@dataclass
class Approval:
    granted: bool
    by: Optional[str] = None          # who answered, as the dashboard reports it
    note: Optional[str] = None        # their note, or why no one could answer
    amount_usd: Optional[float] = None  # the amount approved, if one was asked for
    approval_id: Optional[str] = None


_asked: dict = {}  # approval_id -> (perf_counter at request, reason, amount_usd)


def _ask(reason: str, amount_usd: Optional[float], timeout_s: float):
    """Send the request; returns (run, approval_id) to wait on, or an Approval already decided."""
    run = get_active_run()
    if run is None:
        return Approval(False, note="request_approval() was called outside a governed run")
    if control._client is None:
        return Approval(False, note="Local mode: there is no one to approve this request")
    if not control._supported["approvals"]:
        return Approval(False, note="This Fences server doesn't support approvals")
    approval_id = str(uuid.uuid4())
    _asked[approval_id] = (time.perf_counter(), reason, amount_usd)
    resp = control._client.request_approval(run.run_id, approval_id, reason, amount_usd, timeout_s)  # so the server expires it in step
    denied = None
    if resp.get("unsupported"):
        control.unsupported("approvals", "approval requests", "they are denied")
        denied = "This Fences server doesn't support approvals"
    elif "network_error" in resp:
        denied = "Couldn't reach Fences to ask for approval"
    elif resp.get("ok") is False:  # rejected: no one will be asked, so don't wait out the timeout
        denied = f"Fences didn't accept the approval request (HTTP {resp.get('status')})"
    if denied:
        _asked.pop(approval_id, None)
        return Approval(False, note=denied)
    run.approvals[approval_id] = None
    run.awaiting += 1
    control._wake.set()  # poll fast while someone decides
    return run, approval_id


def _waiting(run, approval_id: str, deadline: float) -> bool:
    return run.approvals.get(approval_id) is None and time.monotonic() < deadline


def _answer(run, approval_id: str) -> Approval:
    run.awaiting -= 1
    c = run.approvals.pop(approval_id, None)  # the dashboard's approval command, or None: no answer in time
    amount = c.get("amount_usd") if c else None  # granted only when it's true, and an amount only when it's a real one
    amount = amount if control.is_number(amount) and amount > 0 else None
    answer = (Approval(c.get("granted") is True, by=c.get("by"), note=c.get("note"), amount_usd=amount, approval_id=approval_id)
              if c else Approval(False, note="No answer before the timeout", approval_id=approval_id))
    if answer.granted and answer.amount_usd and run.budget_usd is not None:  # an unlimited run stays unlimited
        run.budget_usd += answer.amount_usd
    started, reason, asked_usd = _asked.pop(approval_id, (time.perf_counter(), None, None))
    record_span("fences.approval", time.perf_counter() - started, {  # one span from request to answer
        "fences.approval.id": approval_id, "fences.approval.reason": reason,
        "fences.approval.amount_usd": asked_usd, "fences.approval.granted": answer.granted,
        "fences.approval.by": answer.by, "fences.approval.note": answer.note,
        "fences.approval.granted_usd": answer.amount_usd})
    return answer


def request_approval_sync(reason: str, amount_usd: Optional[float] = None, timeout_s: float = 600) -> Approval:
    """request_approval() for synchronous agents. Same arguments and result."""
    asked = _ask(reason, amount_usd, timeout_s)
    if isinstance(asked, Approval):
        return asked
    run, approval_id = asked
    deadline = time.monotonic() + timeout_s
    while _waiting(run, approval_id, deadline):
        time.sleep(0.1)
    return _answer(run, approval_id)


async def request_approval(reason: str, amount_usd: Optional[float] = None, timeout_s: float = 600) -> Approval:
    """
    Ask a person, through the Fences dashboard, to approve something, and wait for the answer:

        approval = await agentfences.request_approval("Refund $240 to order 1182?", amount_usd=240)
        if not approval.granted:
            return f"I didn't issue the refund: {approval.note}"

    If an amount is approved, the run's budget goes up by that much. Returns denied right
    away in local mode, outside a run, or when the server can't take requests, and denied
    after timeout_s seconds with no answer: it never hangs and never raises.
    """
    asked = await asyncio.to_thread(_ask, reason, amount_usd, timeout_s)
    if isinstance(asked, Approval):
        return asked
    run, approval_id = asked
    deadline = time.monotonic() + timeout_s
    while _waiting(run, approval_id, deadline):
        await asyncio.sleep(0.1)
    return _answer(run, approval_id)
