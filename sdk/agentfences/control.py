"""
Live control from the Fences dashboard: a heartbeat that reports live runs and brings
back commands (stop, pause, resume, limits, approval answers), pausing at checkpoints,
and approval requests. Cloud mode only; an older server without these endpoints gets
one warning and runs carry on as before.
"""
import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("agentfences")

FAST_S = 2.0  # heartbeat interval while a run is paused or waiting for an approval

_live: dict = {}               # run_id -> RunState, for runs in progress
_lock = threading.Lock()
_wake = threading.Event()      # nudges the heartbeat to report right away
_thread: Optional[threading.Thread] = None
_client = None
_interval = 15.0
pause_timeout_s = 3600.0
_supported = {"heartbeat": True, "approvals": True}
_asked: dict = {}              # approval_id -> (perf_counter at request, reason, amount_usd)


@dataclass
class Approval:
    granted: bool
    by: Optional[str] = None          # who answered, as the dashboard reports it
    note: Optional[str] = None        # their note, or why no one could answer
    amount_usd: Optional[float] = None  # the amount approved, if one was asked for


def configure(client, heartbeat_s: float, pause_timeout: float):
    global _client, _interval, pause_timeout_s, _thread
    _client, _interval, pause_timeout_s = client, heartbeat_s, pause_timeout
    _supported.update(heartbeat=True, approvals=True)
    _wake.set()  # a running heartbeat picks up the new client and interval now, not after its old wait
    if client is not None and _thread is None:
        _thread = threading.Thread(target=_heartbeat, name="agentfences-heartbeat", daemon=True)
        _thread.start()


def register(run):
    with _lock:
        _live[run.run_id] = run


def unregister(run):
    with _lock:
        _live.pop(run.run_id, None)


def _unsupported(kind: str, what: str, consequence: str):
    if _supported[kind]:
        _supported[kind] = False
        log.warning("agentfences: this Fences server doesn't support %s yet, so %s", what, consequence)


# ── Commands ──────────────────────────────────────────────────────────────────

LIMIT_FIELDS = ("budget_usd", "max_iterations", "max_duration_ms", "max_tokens")


def apply(run, commands):
    """Apply the commands addressed to this run (from a heartbeat or a checkpoint response)."""
    for c in commands or ():
        if c.get("run_id") not in (None, run.run_id):
            continue
        kind = c.get("type")
        if kind == "stop":
            run.stopped = True
            _resume(run)
        elif kind == "pause":
            run.paused = True
            run.resumed.clear()
            _wake.set()
        elif kind == "resume":
            _resume(run)
        elif kind == "limits":
            for f in LIMIT_FIELDS:
                if c.get(f) is not None:
                    setattr(run, f, c[f])
        elif kind == "approval" and c.get("approval_id") in run.approvals:
            run.approvals[c["approval_id"]] = Approval(
                granted=bool(c.get("granted")), by=c.get("by"), note=c.get("note"), amount_usd=c.get("amount_usd"))


def _resume(run):
    run.paused = False
    run.resumed.set()


def dispatch(commands):
    with _lock:
        runs = dict(_live)
    for run_id in {c.get("run_id") for c in commands or ()}:
        if run_id in runs:
            apply(runs[run_id], commands)


def _heartbeat():
    while True:
        with _lock:
            busy = any(r.paused or r.awaiting for r in _live.values())
        _wake.wait(FAST_S if busy else _interval)
        _wake.clear()
        with _lock:
            runs = list(_live.values())
        if not runs or _client is None:
            continue
        try:
            resp = _client.heartbeat([{"run_id": r.run_id, "iterations": r.iterations, "spent_usd": r.cost_usd,
                                       "tokens_used": r.tokens_used, "paused": r.paused} for r in runs])
        except Exception:
            continue
        if resp.get("unsupported"):
            _unsupported("heartbeat", "the heartbeat", "a stop from the dashboard applies at the next checkpoint and pause isn't available")
            return
        dispatch(resp.get("commands"))


# ── Pausing ───────────────────────────────────────────────────────────────────

def _pause_over(run, deadline: float) -> bool:
    if not run.paused:
        return True
    if time.monotonic() >= deadline:
        run.paused, run.pause_expired = False, True
        return True
    return False


def wait_if_paused(run):
    """Block a synchronous agent at its checkpoint while the run is paused."""
    deadline = time.monotonic() + pause_timeout_s
    while not _pause_over(run, deadline):
        run.resumed.wait(min(1.0, max(0.0, deadline - time.monotonic())))


async def wait_if_paused_async(run):
    """The same for async agents, without tying up a thread per paused run."""
    deadline = time.monotonic() + pause_timeout_s
    while not _pause_over(run, deadline):
        await asyncio.sleep(0.2)


# ── Approvals ─────────────────────────────────────────────────────────────────

def _ask(reason: str, amount_usd: Optional[float]):
    """Send the request; returns (run, approval_id) to wait on, or an Approval already decided."""
    from .core import get_active_run
    run = get_active_run()
    if run is None:
        return Approval(False, note="request_approval() was called outside a governed run")
    if _client is None:
        return Approval(False, note="Local mode: there is no one to approve this request")
    if not _supported["approvals"]:
        return Approval(False, note="This Fences server doesn't support approvals")
    approval_id = str(uuid.uuid4())
    _asked[approval_id] = (time.perf_counter(), reason, amount_usd)
    resp = _client.request_approval(run.run_id, approval_id, reason, amount_usd)
    if resp.get("unsupported") or "network_error" in resp:
        _asked.pop(approval_id, None)
    if resp.get("unsupported"):
        _unsupported("approvals", "approval requests", "they are denied")
        return Approval(False, note="This Fences server doesn't support approvals")
    if "network_error" in resp:
        return Approval(False, note="Couldn't reach Fences to ask for approval")
    run.approvals[approval_id] = None
    run.awaiting += 1
    _wake.set()  # poll fast while someone decides
    return run, approval_id


def _waiting(run, approval_id: str, deadline: float) -> bool:
    return run.approvals.get(approval_id) is None and time.monotonic() < deadline


def _settle(run, approval_id: str) -> Approval:
    from .core import record_span
    run.awaiting -= 1
    answer = run.approvals.pop(approval_id, None) or Approval(False, note="No answer before the timeout")
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
    asked = _ask(reason, amount_usd)
    if isinstance(asked, Approval):
        return asked
    run, approval_id = asked
    deadline = time.monotonic() + timeout_s
    while _waiting(run, approval_id, deadline):
        time.sleep(0.1)
    return _settle(run, approval_id)


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
    asked = await asyncio.to_thread(_ask, reason, amount_usd)
    if isinstance(asked, Approval):
        return asked
    run, approval_id = asked
    deadline = time.monotonic() + timeout_s
    while _waiting(run, approval_id, deadline):
        await asyncio.sleep(0.1)
    return _settle(run, approval_id)
