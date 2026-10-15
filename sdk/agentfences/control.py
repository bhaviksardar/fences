"""
Live control from the Fences dashboard: a heartbeat that reports live runs and brings
back commands (stop, pause, resume, quarantine, limits, approval answers) and
acknowledges them, containment hooks, pausing at checkpoints, and approval requests.
Cloud mode only; an older server without these endpoints gets one warning and runs
carry on as before.
"""
import asyncio
import atexit
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("agentfences")

FAST_S = 2.0  # heartbeat interval while a run is paused or waiting for an approval
RETRY = {429, *range(500, 600)}  # heartbeat statuses whose acks go again with the next heartbeat
HOOK_TIMEOUT_S = 30.0  # a hook still running after this is reported as timed out

_live: dict = {}               # run_id -> RunState, for runs in progress
_lock = threading.Lock()
_wake = threading.Event()      # nudges the heartbeat to report right away
_thread: Optional[threading.Thread] = None
_client = None
_interval = 15.0
pause_timeout_s = 3600.0
_stop_grace_s: Optional[float] = 30.0
_supported = {"heartbeat": True, "approvals": True, "guard": True}
_asked: dict = {}              # approval_id -> (perf_counter at request, reason, amount_usd)
_hooks: dict = {"stop": None, "quarantine": None}  # init(on_stop=, on_quarantine=)
_acks: list = []               # acknowledgements waiting for the next heartbeat
_hooked: set = set()           # quarantine ids whose hook has run in this process
_running_hooks: list = []      # hook threads still going, waited for at exit


@dataclass
class Approval:
    granted: bool
    by: Optional[str] = None          # who answered, as the dashboard reports it
    note: Optional[str] = None        # their note, or why no one could answer
    amount_usd: Optional[float] = None  # the amount approved, if one was asked for
    approval_id: Optional[str] = None


def configure(client, heartbeat_s: float, pause_timeout: float, stop_grace_s: Optional[float] = 30.0,
              on_stop: Optional[Callable] = None, on_quarantine: Optional[Callable] = None):
    global _client, _interval, pause_timeout_s, _stop_grace_s, _thread
    _client, _interval, pause_timeout_s, _stop_grace_s = client, heartbeat_s, pause_timeout, stop_grace_s
    _hooks.update(stop=on_stop, quarantine=on_quarantine)
    _supported.update(heartbeat=True, approvals=True, guard=True)
    _wake.set()  # a running heartbeat picks up the new client and interval now, not after its old wait
    if client is not None and (_thread is None or not _thread.is_alive()):  # it ends on a server without the heartbeat
        _thread = threading.Thread(target=_heartbeat, name="agentfences-heartbeat", daemon=True)
        _thread.start()


def register(run):
    with _lock:
        _live[run.run_id] = run


def unregister(run):
    with _lock:
        _live.pop(run.run_id, None)


def unsupported(kind: str, what: str, consequence: str):
    if _supported[kind]:
        _supported[kind] = False
        log.warning("agentfences: this Fences server doesn't support %s yet, so %s", what, consequence)


# ── Commands ──────────────────────────────────────────────────────────────────

LIMIT_FIELDS = ("budget_usd", "max_iterations", "max_duration_ms", "max_tokens")


def apply(run, commands):
    """Apply the commands addressed to this run (from a heartbeat or a checkpoint response), and acknowledge them."""
    # Limits first: a resume in the same reply may release a call held for budget, which re-checks against them at once
    for c in sorted(commands or (), key=lambda c: c.get("type") != "limits"):
        if c.get("run_id") not in (None, run.run_id):
            continue
        kind = c.get("type")
        if kind == "stop":
            if not run.stopped:
                run.stopped, run.stopped_at = True, time.monotonic()
                _wake.set()  # watch for the agent to see it, or cancel it
                _hook("stop", run, c)  # once per run
            _resume(run)
        elif kind in ("pause", "quarantine"):  # a quarantined agent's runs hold at their next checkpoint
            if not run.paused:  # only a change wakes the heartbeat: a quarantine is re-sent on every reply while it lasts
                run.paused = True
                run.resumed.clear()
                _wake.set()
            if kind == "quarantine":
                run.quarantined = True  # guarded tools are blocked here too, even if the server can't be reached
                _hook("quarantine", run, c)
        elif kind == "resume":
            _resume(run)
        elif kind == "limits":
            for f in LIMIT_FIELDS:
                if c.get(f) is not None:
                    setattr(run, f, c[f])
        elif kind == "approval":
            if c.get("approval_id") in run.approvals:
                run.approvals[c["approval_id"]] = Approval(
                    granted=bool(c.get("granted")), by=c.get("by"), note=c.get("note"), amount_usd=c.get("amount_usd"))
        else:
            continue  # unknown: ignored, and not acknowledged
        if c.get("id") and c["id"] not in run.acked:
            run.acked.add(c["id"])
            _ack({"run_id": run.run_id, "id": c["id"], "type": kind, "received_at": time.time()})


def _resume(run):
    run.paused = run.quarantined = False  # releasing a quarantine resumes its runs
    run.resumed.set()


# ── Acknowledgements and containment hooks ────────────────────────────────────

def _ack(entry: dict):
    with _lock:
        _acks.append(entry)
    _wake.set()  # send it now, not at the next interval


def _hook(kind: str, run, c: dict):
    """Run init(on_stop=/on_quarantine=) in its own thread; a quarantine's hook runs once per process."""
    fn = _hooks.get(kind)
    if fn is None:
        return
    if kind == "quarantine":
        key = c.get("id") or ("run", run.run_id)  # every server that sends quarantine gives it an id
        with _lock:
            if key in _hooked:
                return
            _hooked.add(key)
    event = {"type": kind, "run_id": run.run_id, "agent_name": run.agent_name, "id": c.get("id"),
             "by": c.get("by"), "reason": c.get("reason")}
    t = threading.Thread(target=_run_hook, args=(fn, event), name=f"agentfences-{kind}-hook", daemon=True)
    with _lock:
        _running_hooks.append(t)
    t.start()


def _run_hook(fn: Callable, event: dict):
    from .events import error_text
    box: dict = {}

    def call():
        try:
            fn(event)
        except BaseException as e:
            box["error"] = e
    t = threading.Thread(target=call, name=f"agentfences-{event['type']}-hook-call", daemon=True)
    t.start()
    t.join(HOOK_TIMEOUT_S)  # ponytail: a hung hook can't be killed; it's reported and left running
    if t.is_alive():
        result, detail = "timeout", None
    elif "error" in box:
        result, detail = "failed", error_text(box["error"])[:300]
    else:
        result, detail = "ok", None
    if result != "ok":
        log.warning("agentfences: the on_%s hook %s%s", event["type"], "timed out" if result == "timeout" else "failed: ", detail or "")
    if event["id"]:
        _ack({"run_id": event["run_id"], "id": event["id"], "type": event["type"], "hook": result,
              **({"detail": detail} if detail else {}), "finished_at": time.time()})
    with _lock:
        _running_hooks.remove(threading.current_thread())


def drain(timeout: float = HOOK_TIMEOUT_S):
    """
    Wait for running hooks, then send the acknowledgements still waiting. Runs at exit and
    from flush(), so a script that ends right after a stop still finishes its containment,
    and says so, instead of the hook being killed with the process.
    """
    deadline = time.monotonic() + timeout
    with _lock:
        hooks = list(_running_hooks)
    for t in hooks:
        t.join(max(0.0, deadline - time.monotonic()))
    with _lock:
        acks, _acks[:] = list(_acks), []
    if acks and _client is not None and _supported["heartbeat"]:
        try:
            _client.heartbeat([], acks)
        except Exception:
            pass


atexit.register(drain)


def dispatch(commands):
    with _lock:
        runs = dict(_live)
    for run_id in {c.get("run_id") for c in commands or ()}:
        if run_id in runs:
            apply(runs[run_id], commands)


def _unseen_stop(run) -> bool:
    return run.stopped and not run.stop_seen and run.task is not None and not run.cancelled


def _cancel(run):
    """On the run's event loop: cancel a run that hasn't seen its stop. core turns it into FencesStop."""
    if _unseen_stop(run):
        run.cancelled = True
        run.task.cancel()


def cancel_overdue(runs):
    """Cancel async runs that haven't reached a checkpoint within stop_grace_s of a stop."""
    if _stop_grace_s is None:
        return
    for r in runs:
        if _unseen_stop(r) and time.monotonic() - r.stopped_at >= _stop_grace_s:
            try:
                r.loop.call_soon_threadsafe(_cancel, r)
            except RuntimeError:  # the loop has closed: the run is over anyway
                pass


def _heartbeat():
    while True:
        with _lock:
            busy = any(r.paused or r.awaiting or _unseen_stop(r) for r in _live.values())
        _wake.wait(min(FAST_S, _interval, _stop_grace_s or FAST_S) if busy else _interval)  # faster while busy, never slower
        _wake.clear()
        with _lock:
            runs = list(_live.values())
            acks, _acks[:] = list(_acks), []
        cancel_overdue(runs)
        if not (runs or acks) or _client is None:
            continue
        try:
            resp = _client.heartbeat([{"run_id": r.run_id, "iterations": r.iterations, "spent_usd": r.cost_usd,
                                       "tokens_used": r.tokens_used, "paused": r.paused, "stopped": r.stopped,
                                       **({"held": r.held} if r.held else {})} for r in runs],
                                     acks)
        except Exception:
            resp = {"network_error": "heartbeat failed"}
        if "network_error" in resp or resp.get("ok") is False:
            if "network_error" in resp or resp.get("status", 500) in RETRY:  # any other 4xx won't get better by resending
                with _lock:
                    _acks[:0] = acks  # at least once: they go with the next heartbeat
            continue
        if resp.get("unsupported"):
            unsupported("heartbeat", "the heartbeat", "a stop from the dashboard applies at the next checkpoint and pause isn't available")
            return  # stops then arrive only in checkpoint replies, which the agent sees
        dispatch(resp.get("commands"))


# ── Waiting at a limit ────────────────────────────────────────────────────────

# Breaches a person can lift by raising a limit, so on_breach="hold" waits at them
HOLDABLE = {"budget_exceeded", "iteration_limit", "time_limit", "token_limit", "key_daily_budget", "key_monthly_budget",
            "limit_reached"}
BUDGET_BREACHES = {"budget_exceeded", "key_daily_budget", "key_monthly_budget"}


def holds(run) -> bool:
    """Whether this run waits at a limit: on_breach="hold", in cloud mode (someone can raise it), and not stopped."""
    return run.on_breach == "hold" and _client is not None and not run.stopped


def start_hold(run, kind: str):
    """Pause the run at a limit: kind "breach" (a checkpoint breached) or "budget" (a call refused for budget)."""
    run.held = kind
    if not run.paused:
        run.paused = True
        run.resumed.clear()
        _wake.set()


def end_hold(run) -> str:
    """After waiting: "released" (an operator raised the limit), "stopped", or "timeout"."""
    run.held = None
    if run.stopped:
        return "stopped"
    if run.pause_expired:
        run.pause_expired = False  # the run ends with the limit it was held at, not "paused"
        return "timeout"
    return "released"


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

def _ask(reason: str, amount_usd: Optional[float], timeout_s: float):
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
    resp = _client.request_approval(run.run_id, approval_id, reason, amount_usd, timeout_s)  # so the server expires it in step
    if resp.get("unsupported") or "network_error" in resp:
        _asked.pop(approval_id, None)
    if resp.get("unsupported"):
        unsupported("approvals", "approval requests", "they are denied")
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
    answer.approval_id = approval_id
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
    asked = await asyncio.to_thread(_ask, reason, amount_usd, timeout_s)
    if isinstance(asked, Approval):
        return asked
    run, approval_id = asked
    deadline = time.monotonic() + timeout_s
    while _waiting(run, approval_id, deadline):
        await asyncio.sleep(0.1)
    return _settle(run, approval_id)
