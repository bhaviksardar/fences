"""
Budget prechecks: before a model call goes out, work out the most it could cost and
refuse it (FencesStop) if that could take the run past its budget. The amount is held
while the call is in flight, so parallel calls can't overspend together, and released at
the checkpoint that records the call's actual cost. Used by instrument=True and the
framework integrations, which see each request before it's sent.
"""
import json
import logging
import re
from typing import Optional

from . import control
from .results import breach_result
from .run import get_active_run, reserve_lock
from .exceptions import FencesStop
from .pricing import price_of, cost_of, read_usage

log = logging.getLogger("agentfences")

DATA_URI = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=\\]+")
FILE_TOKENS = 2_000       # ponytail: a flat guess per embedded image or file; providers price them by size
BYTES_PER_TOKEN = 2       # ponytail: tokenizers average 3-4 bytes a token for English and code; 2 keeps the estimate high
DEFAULT_MAX_OUT = 32_768  # for the few models whose output cap isn't in the price table

MAX_OUT_PARAMS = ("max_completion_tokens", "max_output_tokens", "max_tokens")


def max_out(params: dict) -> Optional[int]:
    """The output-token cap a request sets, if any."""
    return next((params[k] for k in MAX_OUT_PARAMS if type(params.get(k)) is int and params[k] > 0), None)


def worst_case_usd(model: Optional[str], request, out_cap: Optional[int] = None, n: int = 1) -> Optional[float]:
    """
    The most a call could cost: its input, estimated high from the request's size, at the
    dearest input rate, plus its output cap (or the model's) at the output rate. None for
    a model with no known price, which can't be prechecked.
    """
    p = price_of(model) if model else None
    if p is None:
        return None
    text = json.dumps(request, default=str)
    files = len(DATA_URI.findall(text))
    tokens_in = len(DATA_URI.sub("", text).encode()) // BYTES_PER_TOKEN + files * FILE_TOKENS
    tokens_out = (out_cap or p.get("max_out") or DEFAULT_MAX_OUT) * max(n, 1)
    rates = dict(p, **p["tier"]) if "above" in p and tokens_in > p["above"] else p
    return tokens_in * max(rates["in"], rates.get("cache_write", 0)) + tokens_out * rates["out"]


def _refuse(run, worst: float, out_cap: Optional[int]):
    result = breach_result("budget_exceeded", spent=run.cost_usd, limit=run.budget_usd)
    room = max(run.budget_usd - run.cost_usd - run.reserved_usd, 0)
    result.spent_usd = run.cost_usd
    result.message = (f"I can't make another model call without risking going over my budget (spent ${run.cost_usd:.4f} "
                      f"of ${run.budget_usd:.4f}; the next call could cost up to ${worst:.4f}). I've stopped here.")
    result.system_prompt = ("The run's budget can't cover another model call. Stop now; don't call any more tools or models.")
    run.last_breach = "budget_exceeded"  # the run ends breached even if the agent catches FencesStop and returns
    log.warning("agentfences: %s: refused a model call that could cost up to $%.4f with $%.4f of its budget left.%s",
                run.agent_name, worst, room, "" if out_cap else " Setting max_tokens on the call lowers its worst case.")
    raise FencesStop(result)


def _prepare(model: Optional[str], request, params: Optional[dict]):
    """(run, worst case, output cap) for a call to check, or None when there's nothing to check."""
    run = get_active_run()
    if run is not None and run.cancelled:  # halted after a stop: work it left running can't call the model
        from .core import _result
        raise FencesStop(_result(run))
    if run is None or run.budget_usd is None:
        return None
    params = params or {}
    out_cap = max_out(params)
    worst = worst_case_usd(model, request, out_cap, params.get("n") or 1)
    return None if worst is None else (run, worst, out_cap)


def _take(run, worst: float) -> Optional[list]:
    """Hold worst if it fits in what's left of the budget; None if it doesn't."""
    with reserve_lock:
        if round(run.cost_usd + run.reserved_usd + worst, 9) > run.budget_usd:
            return None
        h = [worst, False, None]  # amount, finished, the response that priced it
        run.holds.append(h)
        run.reserved_usd += worst
        return h


def _gave_up(run, worst: float, out_cap: Optional[int]):
    """A call that doesn't fit, and won't wait (or waited and wasn't released): FencesStop."""
    if run.stopped:
        from .core import _result
        raise FencesStop(_result(run))
    _refuse(run, worst, out_cap)


def hold(model: Optional[str], request, params: Optional[dict] = None) -> Optional[list]:
    """
    Before a model call: raise FencesStop if its worst case doesn't fit in what's left of the
    run's budget, else hold that amount. Returns the hold to pass to finish(), or None when
    there's nothing to check (no run, no budget, or a model with no known price). With
    on_breach="hold" a call that doesn't fit waits for someone to raise the budget first.
    """
    prepared = _prepare(model, request, params)
    if prepared is None:
        return None
    run, worst, out_cap = prepared
    while (h := _take(run, worst)) is None:
        if not control.holds(run):
            _gave_up(run, worst, out_cap)
        control.start_hold(run, "budget")
        control.wait_if_paused(run)
        if control.end_hold(run) != "released":
            _gave_up(run, worst, out_cap)
    return h


async def hold_async(model: Optional[str], request, params: Optional[dict] = None) -> Optional[list]:
    """hold() for async callers: waiting for a raised budget doesn't block the event loop."""
    prepared = _prepare(model, request, params)
    if prepared is None:
        return None
    run, worst, out_cap = prepared
    while (h := _take(run, worst)) is None:
        if not control.holds(run):
            _gave_up(run, worst, out_cap)
        control.start_hold(run, "budget")
        await control.wait_if_paused_async(run)
        if control.end_hold(run) != "released":
            _gave_up(run, worst, out_cap)
    return h


def finish(h: Optional[list], response=None, failed: bool = False):
    """
    After the call. A failed call isn't billed, so its hold is released now. Otherwise the
    hold shrinks to the response's actual cost, and checkpoint(response), which records that
    cost, releases it; until then it still counts against the budget. A call with no cost to
    read (a stream without usage) keeps its worst case until the next checkpoint.
    """
    if h is None:
        return
    run = get_active_run()
    actual = cost_of(response)[0] if response is not None and read_usage(response) is not None else None
    with reserve_lock:
        if run is None or not any(x is h for x in run.holds):
            return
        if failed:
            run.holds[:] = [x for x in run.holds if x is not h]  # by identity: parallel calls can hold equal amounts
            run.reserved_usd -= h[0]
            return
        if actual is not None:
            run.reserved_usd += actual - h[0]
            h[0], h[2] = actual, response
        h[1] = True
