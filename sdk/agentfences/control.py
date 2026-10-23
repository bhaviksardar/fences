"""
Live control from the Fences dashboard: a heartbeat that reports live runs and brings
back commands (stop, pause, resume, quarantine, limits, approval answers) and
acknowledges them, containment hooks, waiting at a limit, and pausing at checkpoints.
Cloud mode only; an older server without these endpoints gets one warning and runs
carry on as before.
"""
import asyncio
import atexit
import ctypes
import logging
import math
import random
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("agentfences")

FAST_S = 2.0  # heartbeat interval while a run is paused or waiting for an approval
RETRY = {429, *range(500, 600)}  # heartbeat statuses whose acks go again with the next heartbeat
HOOK_TIMEOUT_S = 30.0  # a hook still running after this is reported as timed out
MAX_BACKOFF_S = 60.0   # longest wait between heartbeats after failures (jittered)

_live: dict = {}               # run_id -> RunState, for runs in progress
_lock = threading.Lock()
_wake = threading.Event()      # nudges the heartbeat to report right away
_thread: Optional[threading.Thread] = None
_client = None
_interval = 15.0
pause_timeout_s = 3600.0
_stop_grace_s: Optional[float] = 30.0
_supported = {"heartbeat": True, "approvals": True, "guard": True}
_hooks: dict = {"stop": None, "quarantine": None}  # init(on_stop=, on_quarantine=)
_acks: list = []               # acknowledgements waiting for the next heartbeat
_hooked: set = set()           # quarantine ids whose hook has run in this process
_running_hooks: list = []      # hook threads still going, waited for at exit


def configure(client, heartbeat_s: float, pause_timeout: float, stop_grace_s: Optional[float] = 30.0,
              on_stop: Optional[Callable] = None, on_quarantine: Optional[Callable] = None):
    global _client, _interval, pause_timeout_s, _stop_grace_s, _thread
    _client, _interval, pause_timeout_s, _stop_grace_s = client, heartbeat_s, pause_timeout, stop_grace_s
    _hooks.update(stop=on_stop, quarantine=on_quarantine)
    _supported.update(heartbeat=True, approvals=True, guard=True)
    _backoff.update(failed=0, next_try=0.0)
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
TOTAL_FIELDS = (("spent_usd", "cost_usd"), ("iterations", "iterations"), ("tokens_used", "tokens_used"))


def is_number(v) -> bool:
    """A finite int or float: a NaN would switch a limit off, and an infinite one would lift it."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def adopt(run, reply: dict, fields, null_budget: bool = False):
    """Set the run's fields from a server reply: numbers only, so a null or odd value can't break local enforcement.
    fields: names, or (reply name, run attribute) pairs. null_budget: a null budget_usd means unlimited spend."""
    for f in fields:
        src, dst = f if isinstance(f, tuple) else (f, f)
        v = reply.get(src)
        if is_number(v) or (null_budget and src == "budget_usd" and src in reply and v is None):
            setattr(run, dst, v)


def well_formed(commands) -> list:
    """The commands in a reply that can be applied: objects whose ids are strings. A malformed one is
    skipped, so it can't stop the heartbeat or a checkpoint."""
    ok = lambda v: v is None or isinstance(v, str)
    return [c for c in (commands if isinstance(commands, list) else ())
            if isinstance(c, dict) and ok(c.get("run_id")) and ok(c.get("id")) and ok(c.get("approval_id"))]


def apply(run, commands):
    """Apply the commands addressed to this run (from a heartbeat or a checkpoint response), and acknowledge them."""
    # Limits first: a resume in the same reply may release a call held for budget, which re-checks against them at once
    for c in sorted(well_formed(commands), key=lambda c: c.get("type") != "limits"):
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
            adopt(run, c, LIMIT_FIELDS)
        elif kind == "approval":
            if c.get("approval_id") in run.approvals:
                run.approvals[c["approval_id"]] = c  # approvals.py turns it into the answer
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
    for run_id in {c.get("run_id") for c in well_formed(commands)}:
        if run_id in runs:
            apply(runs[run_id], commands)


# ── Halting a run stuck after a stop ──────────────────────────────────────────

class Halt(BaseException):
    """
    Raised inside a sync run's thread when it hasn't seen a stop within stop_grace_s; the
    governed call turns it into FencesStop. A BaseException, so `except Exception` in the
    agent doesn't swallow it.
    """


# CPython's way to raise in another thread; elsewhere (PyPy) a stuck sync run isn't halted
_raise_in = getattr(getattr(ctypes, "pythonapi", None), "PyThreadState_SetAsyncExc", None)
if _raise_in is not None:
    _raise_in.argtypes, _raise_in.restype = [ctypes.c_ulong, ctypes.py_object], ctypes.c_int


def _unseen_stop(run) -> bool:
    return (run.stopped and not run.stop_seen and not run.cancelled
            and (run.task is not None or (run.thread is not None and _raise_in is not None)))


def _cancel(run):
    """On the run's event loop: cancel a run that hasn't seen its stop. core turns it into FencesStop."""
    if _unseen_stop(run) and run.task is not None:
        run.cancelled = True
        run.task.cancel()


def _halt(run):
    """Raise Halt in a sync run's thread, once. It lands at the thread's next Python instruction:
    at once in Python code, or when a blocking call (a socket read, a sleep) returns."""
    with _lock:  # disarm() takes it too: a halt never lands after the run's code is done
        if _unseen_stop(run) and run.task is None and run.thread is not None:
            run.cancelled = True
            _raise_in(run.thread, Halt)


def disarm(run):
    """The run's own code is done: no halt from here on, and one sent but not yet raised is withdrawn,
    unless a run it's nested in was halted too: then the halt is that run's."""
    with _lock:
        thread, run.thread = run.thread, None
        if run.cancelled and thread is not None and not _halted_outer(run, thread):
            _raise_in(thread, ctypes.py_object())  # NULL: clears a pending exception


def halted_outer(run) -> bool:
    """A run this one is nested in, on this thread, was halted too: a Halt here is for it as well."""
    with _lock:
        return _halted_outer(run, threading.get_ident())


def _halted_outer(run, thread) -> bool:  # under _lock
    return any(r is not run and r.thread == thread and r.cancelled and not r.stop_seen for r in _live.values())


def cancel_overdue(runs):
    """Cancel async runs, and halt sync ones, that haven't reached a checkpoint within stop_grace_s of a stop."""
    if _stop_grace_s is None:
        return
    for r in runs:
        if not (_unseen_stop(r) and time.monotonic() - r.stopped_at >= _stop_grace_s):
            continue
        if r.task is None:
            _halt(r)
            continue
        try:
            r.loop.call_soon_threadsafe(_cancel, r)
        except RuntimeError:  # the loop has closed: the run is over anyway
            pass


_backoff = {"failed": 0, "next_try": 0.0}  # heartbeats failed in a row, and when the next may go


def _heartbeat():
    while True:
        with _lock:
            busy = any(r.paused or r.awaiting or _unseen_stop(r) for r in _live.values())
        _wake.wait(min(FAST_S, _interval, _stop_grace_s or FAST_S) if busy else _interval)  # faster while busy, never slower
        _wake.clear()
        with _lock:
            runs = list(_live.values())
        cancel_overdue(runs)  # local, so it keeps its schedule while heartbeats back off
        if not (runs or _acks) or _client is None or time.monotonic() < _backoff["next_try"]:
            continue
        with _lock:
            acks, _acks[:] = list(_acks), []
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
                # Back off, doubling from the shortest interval in use, with jitter, so SDKs coming back
                # from the same outage don't all beat at once
                _backoff["failed"] += 1
                wait = min(min(FAST_S, _interval) * 2 ** (_backoff["failed"] - 1), MAX_BACKOFF_S)
                _backoff["next_try"] = time.monotonic() + wait * random.uniform(0.5, 1.0)
            continue
        _backoff.update(failed=0, next_try=0.0)
        if resp.get("unsupported"):
            unsupported("heartbeat", "the heartbeat", "a stop from the dashboard applies at the next checkpoint and pause isn't available")
            return  # stops then arrive only in checkpoint replies, which the agent sees
        try:
            dispatch(resp.get("commands"))
        except Exception:  # the heartbeat carries stop and halt for every run: nothing in a reply may end it
            log.exception("agentfences: couldn't apply a heartbeat reply's commands")


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
