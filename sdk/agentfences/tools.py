"""
@tool: record each call of an agent's tool (name, a short summary of its arguments,
success or failure, and latency) as evidence on the active run. Optionally give it a
timeout, and guard it: ask Fences before it runs, reserving what it may cost.
"""
import asyncio
import contextvars
import functools
import inspect
import reprlib
import threading
import time
import uuid
from typing import Callable, Optional, Union

from . import approvals, control, events
from .core import check_amount, get_active_run, _local_breach, reserve_lock as _reserve_lock
from .exceptions import ToolBlocked
from .spans import MAX_ARGS, record_tool_call

_short = reprlib.Repr()
_short.maxstring, _short.maxother, _short.maxlist, _short.maxdict = 60, 60, 5, 5


def _items(func: Callable, args: tuple, kwargs: dict) -> list:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return [(k, v) for k, v in bound.arguments.items() if k not in ("self", "cls")]
    except (TypeError, ValueError):  # builtins or odd signatures: fall back to positions
        return [(str(i), v) for i, v in enumerate(args)] + list(kwargs.items())


def args_summary(func: Callable, args: tuple, kwargs: dict) -> str:
    """`query='cats', limit=5`, each value shortened, the whole thing capped."""
    return items_summary(_items(func, args, kwargs))


def arg_names(func: Callable, args: tuple, kwargs: dict) -> str:
    """`query, limit`: what a tool call span carries by default (init(tool_arguments=...))."""
    return ", ".join(k for k, _ in _items(func, args, kwargs))


def items_summary(items) -> str:
    text = ", ".join(f"{k}={_short.repr(v)}" for k, v in items)
    return text if len(text) <= MAX_ARGS else text[:MAX_ARGS - 1] + "…"


# ── Timeouts ──────────────────────────────────────────────────────────────────

async def _timed_async(coro, timeout_s: Optional[float], label: str):
    if timeout_s is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout_s)
    except asyncio.TimeoutError:  # a different class from TimeoutError before Python 3.11
        raise TimeoutError(f"{label} took longer than {timeout_s}s") from None


def _timed_sync(f: Callable, args: tuple, kwargs: dict, timeout_s: Optional[float], label: str):
    if timeout_s is None:
        return f(*args, **kwargs)
    box: dict = {}
    ctx = contextvars.copy_context()  # the tool still sees the active run

    def call():
        try:
            box["out"] = ctx.run(f, *args, **kwargs)
        except BaseException as e:
            box["err"] = e
    # ponytail: Python can't kill a thread, so a timed-out sync tool finishes in the background; async tools are cancelled
    t = threading.Thread(target=call, name=f"agentfences-tool-{label}", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{label} took longer than {timeout_s}s")
    if "err" in box:
        raise box["err"]
    return box["out"]


# ── Guards ────────────────────────────────────────────────────────────────────

def _block(run, label: str, reason: str, breach: Optional[str] = None):
    if run.stopped:
        run.stop_seen = True  # the agent learns of the stop here, so don't cancel it
    raise ToolBlocked(label, reason, breach)


def _local_check(run, amount: float):
    """(reason, breach) when the run can't take this call, else None. Hold _reserve_lock."""
    if run.stopped:
        return "the run was stopped from the Fences dashboard", "stopped_by_user"
    breach = run.last_breach or _local_breach(run)
    if breach:
        return f"the run is over a limit ({breach})", breach
    if run.budget_usd is not None and round(run.cost_usd + run.reserved_usd + amount, 9) > run.budget_usd:
        return (f"${amount:.2f} more would take the run past its ${run.budget_usd:.2f} budget "
                f"(${run.cost_usd:.2f} spent, ${run.reserved_usd:.2f} held by running tools)"), "budget_exceeded"
    return None


def _approve(run, label: str, summary: str, amount: float) -> str:
    answer = approvals.request_approval_sync(f"{label}({summary})", amount_usd=amount or None)
    if not answer.granted:
        _block(run, label, f"not approved: {answer.note}" if answer.note else "not approved")
    return answer.approval_id


def _open(run, label: str, summary: str, amount: float, approve: bool) -> str:
    """Check a guarded call and reserve its amount; returns the guard id, or raises ToolBlocked."""
    from . import core
    if run.quarantined:  # before waiting out the pause a quarantine brings: blocked now, not when it's released
        _block(run, label, "the agent is quarantined in Fences")
    if run.paused:
        control.wait_if_paused(run)
    if run.pause_expired:
        _block(run, label, "the run was paused from the Fences dashboard and not resumed in time", "paused")
    approval_id = _approve(run, label, summary, amount) if approve else None
    client = core._client
    if client is None:  # local mode: the same rules, applied here
        with _reserve_lock:
            denied = _local_check(run, amount)
            if not denied:
                run.reserved_usd += amount
        if denied:
            _block(run, label, *denied)
        return str(uuid.uuid4())
    guard_id = str(uuid.uuid4())  # one per call: the server answers a re-check of a denied id afresh
    while True:
        try:
            return _check_with_server(run, client, label, summary, amount, approval_id, guard_id)
        except ToolBlocked as e:
            if e.breach_type not in control.BUDGET_BREACHES or not control.holds(run):
                raise
            control.start_hold(run, "budget")  # on_breach="hold": wait for someone to raise the budget, then check again
            control.wait_if_paused(run)
            if control.end_hold(run) != "released":
                if run.stopped:
                    _block(run, label, "the run was stopped from the Fences dashboard", "stopped_by_user")
                raise


def _check_with_server(run, client, label: str, summary: str, amount: float, approval_id: Optional[str], guard_id: str) -> str:
    if run.stopped:
        _block(run, label, "the run was stopped from the Fences dashboard", "stopped_by_user")
    with _reserve_lock:  # the server can't see this process's in-flight model calls, so check against them here first
        fits = run.budget_usd is None or round(run.cost_usd + run.reserved_usd + amount, 9) <= run.budget_usd
        if fits:
            run.reserved_usd += amount
    if not fits:
        _block(run, label, f"${amount:.2f} more could take the run past its ${run.budget_usd:.2f} budget", "budget_exceeded")
    try:
        return _ask_server(run, client, label, summary, amount, approval_id, guard_id)
    except BaseException:
        with _reserve_lock:
            run.reserved_usd -= amount
        raise


def _ask_server(run, client, label: str, summary: str, amount: float, approval_id: Optional[str], guard_id: str) -> str:
    sent = events.redact({"type": "guard", "tool": label, "arguments": summary}) or {}  # the check goes ahead without them
    for _ in range(2):  # once more after an approval the server asked for
        resp = client.guard(run.run_id, {"guard_id": guard_id, "phase": "check", "tool": label, "arguments": sent.get("arguments"),
                                         "amount_usd": amount, "approval_id": approval_id})
        control.apply(run, resp.get("commands"))
        if resp.get("allow") is True:
            return guard_id
        if not (resp.get("needs_approval") and approval_id is None):
            break
        approval_id = _approve(run, label, summary, amount)
    if resp.get("unsupported"):
        control.unsupported("guard", "guarded tools", "they are blocked")
        _block(run, label, "this Fences server doesn't support guarded tools")
    if "network_error" in resp or "allow" not in resp:  # no answer: fail closed
        _block(run, label, "Fences rejected the API key" if resp.get("status") in (401, 403) else "couldn't reach Fences to check this call")
    _block(run, label, resp.get("reason") or "not allowed", resp.get("breach"))


def _settle(run, guard_id: str, amount: float, error: Optional[BaseException]):
    """Release the reservation and charge the run: in full unless the tool raised (a timeout or cancel may have done it)."""
    from . import core
    outcome = "ok" if error is None else "unknown" if isinstance(error, (TimeoutError, asyncio.CancelledError)) else "error"
    charged = 0.0 if outcome == "error" else amount
    with _reserve_lock:
        run.reserved_usd -= amount
        run.cost_usd += charged
    if core._client is not None:
        core._client.guard(run.run_id, {"guard_id": guard_id, "phase": "settle", "outcome": outcome, "cost_usd": charged})


async def call_async(run, label: str, summary: str, cost, approve: bool, call: Callable, names: Optional[str] = None):
    """
    Record an async tool call; cost is None/False for an unguarded call, else a function giving its
    amount. summary (the values) goes to the guard and an approval request; the span gets names.
    """
    started, gate = time.perf_counter(), None
    try:
        if cost:
            amount = check_amount(cost(), f"{label}'s cost_usd")  # a NaN or negative amount would let calls past the budget
            gate = (await asyncio.to_thread(_open, run, label, summary, amount, approve), amount)
        out = await call()
    except BaseException as e:
        record_tool_call(label, summary, started, e, arg_names=names)
        if gate:
            await asyncio.to_thread(_settle, run, *gate, e)
        raise
    record_tool_call(label, summary, started, arg_names=names)
    if gate:
        await asyncio.to_thread(_settle, run, *gate, None)
    return out


# ── @tool ─────────────────────────────────────────────────────────────────────

def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, timeout_s: Optional[float] = None,
         guard: bool = False, cost_usd: Union[None, float, Callable[..., float]] = None, approve: bool = False):
    """
    Record every call of an agent tool on the active run:

        @agentfences.tool
        def web_search(query: str) -> list: ...

        @agentfences.tool(name="search", timeout_s=30)
        async def search_docs(q: str): ...

    Records the tool's name, a short summary of its arguments, whether it succeeded (and
    the error if not) and how long it took. Exceptions still reach the caller. Works on
    sync and async functions; outside a governed run it just calls the function.

    timeout_s: raise TimeoutError if the call takes longer. An async tool is cancelled; a
    sync one can't be stopped, so it carries on in a background thread while the agent moves on.

    Guarded tools ask Fences before they run, for actions with consequences:

        @agentfences.tool(cost_usd=lambda order_id, amount: amount, approve=True)
        def issue_refund(order_id: str, amount: float): ...

    guard=True: the tool runs only if the run isn't stopped, paused, over a limit or
    quarantined. Otherwise it raises ToolBlocked and its code never runs; it's blocked
    too if Fences can't be reached (guards fail closed).
    cost_usd: what a call costs the run, a number or a function of the tool's arguments.
    It's reserved before the call, so parallel calls can't overspend the budget together,
    and charged after (unless the tool raised; a timeout is charged, as it may have happened).
    approve=True: a person approves each call from the dashboard first (request_approval).
    Setting cost_usd or approve guards the tool.
    """
    guarded = guard or cost_usd is not None or approve

    def decorate(f: Callable) -> Callable:
        label = name or f.__name__

        def amount(args, kwargs) -> float:
            return (cost_usd(*args, **kwargs) if callable(cost_usd) else cost_usd) or 0

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                run = get_active_run()
                if run is None:
                    return await _timed_async(f(*args, **kwargs), timeout_s, label)
                return await call_async(run, label, args_summary(f, args, kwargs), guarded and (lambda: amount(args, kwargs)),
                                        approve, lambda: _timed_async(f(*args, **kwargs), timeout_s, label),
                                        arg_names(f, args, kwargs))
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            run = get_active_run()
            if run is None:
                return _timed_sync(f, args, kwargs, timeout_s, label)
            summary, names, started, gate = args_summary(f, args, kwargs), arg_names(f, args, kwargs), time.perf_counter(), None
            try:
                if guarded:
                    cost = check_amount(amount(args, kwargs), f"{label}'s cost_usd")
                    gate = (_open(run, label, summary, cost, approve), cost)
                out = _timed_sync(f, args, kwargs, timeout_s, label)
            except BaseException as e:
                record_tool_call(label, summary, started, e, arg_names=names)
                if gate:
                    _settle(run, *gate, e)
                raise
            record_tool_call(label, summary, started, arg_names=names)
            if gate:
                _settle(run, *gate, None)
            return out
        return wrapper

    return decorate(func) if func is not None else decorate
