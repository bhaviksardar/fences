"""
init(), @governed and checkpoint(): starting and ending runs, checking every step against
its limits (the server decides in cloud mode, the same rules apply here otherwise), and
recording decisions and spans.
"""
import asyncio
import atexit
import functools
import inspect
import logging
import threading
import time
import uuid
from typing import Callable, Optional

from . import control, events, outbox
from .client import GovClient
from .exceptions import AgentQuarantined, FencesStop
from .pricing import cost_of, set_custom_prices
from .results import CheckpointResult, breach_result
from .run import RunState, get_active_run, reserve_lock, _set_active_run, _reset_active_run

log = logging.getLogger("agentfences")

_client: Optional[GovClient] = None
_local_only: bool = False
_fail_closed: bool = False
_on_breach: Optional[str] = None


def _check_on_breach(value: Optional[str]) -> Optional[str]:
    if value not in (None, "hold"):
        raise ValueError(f'on_breach must be None or "hold", not {value!r}')
    return value


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:8000",
    local_only: bool = False,
    fail_closed: bool = False,
    prices: Optional[dict] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    redact: Optional[Callable[[dict], Optional[dict]]] = None,
    instrument: bool = False,
    heartbeat_s: float = 15.0,
    pause_timeout_s: float = 3600.0,
    stop_grace_s: Optional[float] = 30.0,
    on_stop: Optional[Callable[[dict], None]] = None,
    on_quarantine: Optional[Callable[[dict], None]] = None,
    on_breach: Optional[str] = None,
    tool_arguments: str = "names",
    allow_http: bool = False,
):
    """
    Initialize Fences. Call once at startup before using @governed.

    Local mode — no backend required:
        agentfences.init(local_only=True)

    Cloud mode — connects to a Fences backend, which is authoritative for limits:
        agentfences.init(api_key="fc_...", endpoint="https://...")

    If the backend can't be reached, limits are enforced locally (fail open).
    Pass fail_closed=True to treat an unreachable backend as a breach instead.

    checkpoint(response) prices most OpenAI, Anthropic, Gemini, Mistral, DeepSeek, xAI
    and Groq models itself. Add or override others in USD per 1M tokens:
        agentfences.init(local_only=True, prices={"my-model": {"input": 1.0, "output": 3.0}})

    environment and release tag every run, e.g. environment="prod", release="v1.4.2".

    redact(event) sees every event before it leaves the process (run start context,
    decisions, run errors) and returns it, changed or not, or None to drop it. If it
    raises, the event is dropped rather than sent unredacted.

    instrument=True records every OpenAI and Anthropic model call made inside a governed
    run (model, tokens, cost, latency, errors) without changing your code.

    In cloud mode a background heartbeat reports live runs every heartbeat_s seconds and
    brings back dashboard commands (stop, pause, resume, limits, approval answers). A paused
    run waits at its next checkpoint, for up to pause_timeout_s seconds. If an async run
    doesn't reach a checkpoint within stop_grace_s seconds of a stop (say, it's stuck in a
    tool), it's cancelled and the governed call raises FencesStop. A sync run is halted the
same way: FencesStop is raised in its thread at once if it's running Python code, or as soon
as a blocking call it's in (a socket read, a sleep) returns (CPython only). A streaming run
(async generator) is cancelled only while its own code runs, never its consumer's. None
never halts.

    on_stop and on_quarantine run in the agent's process, in a background thread, when a
    stop or quarantine arrives from the dashboard, to contain what Fences can't reach (say,
    revoke a credential the agent holds). They get {"type", "run_id", "agent_name", "id",
    "by", "reason"}. on_stop runs once per run, on_quarantine once per quarantine; whether
    each worked, failed or timed out (30 s) is reported to Fences.

    endpoint must use https://, except on this machine (localhost, 127.0.0.1, ::1): over plain
    HTTP anyone on the network path could read the API key and rewrite the server's replies,
    limits included. allow_http=True permits it, for a trusted private network.

    tool_arguments: what a recorded tool call carries of its arguments: "names" (the default:
    `query, limit`), "full" (`query='cats', limit=5`, each value shortened) or "none". A guarded
    tool's check and an approval request still send the values, which rules and approvers need;
    the redact hook can change or drop them.

    on_breach="hold" makes runs wait at a limit instead of ending: when the server says a
    limit is crossed, checkpoint() waits (as if paused) until someone raises it and resumes
    the run, then returns ok and the agent carries on. A model call or guarded tool refused
    for budget waits the same way. A stop ends the wait, and after pause_timeout_s the
    breach is returned as usual. Cloud mode only. Override per agent in @governed.
    """
    global _client, _local_only, _fail_closed, _on_breach
    _on_breach = _check_on_breach(on_breach)
    from . import spans
    if tool_arguments not in spans.TOOL_ARGUMENTS:
        raise ValueError(f"tool_arguments must be one of {spans.TOOL_ARGUMENTS}, not {tool_arguments!r}")
    spans.tool_arguments = tool_arguments
    set_custom_prices(prices)
    events.configure(environment, release, redact)
    if instrument:
        from .instrument import instrument as patch_clients
        patch_clients()
    _local_only = local_only
    _fail_closed = fail_closed

    if local_only:
        _client = None
        control.configure(None, heartbeat_s, pause_timeout_s, stop_grace_s, on_stop, on_quarantine)
        return

    if not api_key:
        raise ValueError(
            "api_key is required unless local_only=True. "
            "Use agentfences.init(local_only=True) for local usage."
        )
    _check_endpoint(endpoint, allow_http)
    _client = GovClient(api_key=api_key, endpoint=endpoint)
    control.configure(_client, heartbeat_s, pause_timeout_s, stop_grace_s, on_stop, on_quarantine)


LOOPBACK = ("localhost", "127.0.0.1", "::1", "[::1]")


def _check_endpoint(endpoint: str, allow_http: bool):
    # Parsed the way requests (urllib3) will parse it, so the host checked is the host connected to;
    # a username or a backslash in it could otherwise make the two disagree
    from urllib3.util import parse_url
    try:
        url = parse_url(endpoint)
    except ValueError:
        url = None
    if url is None or url.scheme not in ("http", "https") or not url.host or url.auth or "\\" in endpoint:
        raise ValueError(f"endpoint must be an https:// URL with no username or backslash, not {endpoint!r}")
    if url.scheme == "https":
        return
    local = url.host.lower() in LOOPBACK
    if not (local or allow_http):
        raise ValueError(
            f"endpoint {endpoint!r} uses plain HTTP: the API key and the server's replies, limits included, "
            "would cross the network unencrypted. Use https://, or pass allow_http=True for a trusted private network.")


def _require_init():
    if not _local_only and _client is None:
        raise RuntimeError(
            "Call agentfences.init() before using Fences. "
            "For local usage: agentfences.init(local_only=True)"
        )


def _warn_unreachable(run: RunState, err: str, ending: bool = False):
    if not run.warned:
        run.warned = True
        log.warning("Fences backend unreachable for run %s (%s); %s", run.run_id, err,
                    "ending the run" if ending else "enforcing limits locally")


# ── Decorator ─────────────────────────────────────────────────────────────────

DEFAULT_LIMITS = {"max_iterations": 100, "max_duration_ms": 300_000, "max_tokens": 0}  # when set nowhere


def governed(
    budget_usd: Optional[float] = None,
    max_iterations: Optional[int] = None,
    max_duration_ms: Optional[int] = None,
    max_tokens: Optional[int] = None,
    on_breach: Optional[str] = None,
):
    """
    Decorator that applies governance policy to an agent function.
    Works on sync functions, async functions and async generators (streaming agents).

    Args:
        budget_usd: Maximum spend allowed for this run in USD.
        max_iterations: Maximum number of checkpoint() calls allowed.
        max_duration_ms: Maximum wall-clock duration in milliseconds.
        max_tokens: Maximum total tokens (input + output) allowed. 0 = no limit.
        on_breach: "hold" to wait at a limit for someone to raise it (see init); defaults to init's.

    Every limit is optional. In cloud mode a limit left out here comes from the agent's
    page in the Fences dashboard. Limits set nowhere default to 100 steps, 5 minutes and
    no token limit, and no budget: the run is unlimited in spend and Fences warns about it.

    checkpoint() never raises: a crossed limit comes back as a CheckpointResult with
    breached=True, so the agent can wrap up gracefully.

    Usage:
        agentfences.init(local_only=True)

        @governed(budget_usd=0.50, max_iterations=20)
        async def run_agent(query: str):
            response = call_llm(query)
            result = await checkpoint(response)
            if result.breached:
                messages.append({"role": "system", "content": result.system_prompt})
                final = call_llm_summarize(messages)
                return final
            return response
    """
    _check_on_breach(on_breach)

    def decorator(func: Callable) -> Callable:
        start = functools.partial(_start_run, func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens, on_breach)

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def agen_wrapper(*args, **kwargs):
                _require_init()
                run = await asyncio.to_thread(start)
                run.loop = asyncio.get_running_loop()
                token, exc, stream = _set_active_run(run), None, func(*args, **kwargs)
                try:
                    while True:
                        # Cancellable after an unseen stop only while the stream's own code runs, never its consumer's
                        run.task = asyncio.current_task()
                        try:
                            item = await stream.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            run.task = None
                        yield item
                except BaseException as e:
                    exc = _stopped(run, e)
                    if exc is not e:
                        raise exc from None
                    raise
                finally:
                    run.task = None
                    _reset_active_run(token)
                    await asyncio.to_thread(_end_run, run, *_outcome(run, exc))
            return agen_wrapper

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _require_init()
                run = await asyncio.to_thread(start)
                run.task, run.loop = asyncio.current_task(), asyncio.get_running_loop()
                token, exc = _set_active_run(run), None
                try:
                    return await func(*args, **kwargs)
                except BaseException as e:
                    exc = _stopped(run, e)
                    if exc is not e:
                        raise exc from None
                    raise
                finally:
                    run.task = None  # first, so a late cancel can't land in the cleanup below
                    _reset_active_run(token)
                    await asyncio.to_thread(_end_run, run, *_outcome(run, exc))
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _require_init()
            run = start()
            token, exc = _set_active_run(run), None
            try:
                try:
                    run.thread = threading.get_ident()  # halted (control.Halt) if a stop goes unseen past stop_grace_s
                    return func(*args, **kwargs)
                finally:
                    control.disarm(run)  # first, so a halt can't land in the cleanup below
            except BaseException as e:
                exc = _stopped(run, e)
                if exc is not e:
                    raise exc from None
                raise
            finally:
                _reset_active_run(token)
                _end_run(run, *_outcome(run, exc))
        return sync_wrapper

    return decorator


def _stopped(run: RunState, e: BaseException) -> BaseException:
    """A cancellation or halt Fences made after an unseen stop becomes FencesStop; anything else is unchanged
    (a halt meant for an outer run passes through an inner one)."""
    if not run.cancelled:
        return e
    if isinstance(e, control.Halt):
        control.disarm(run)  # it may have landed before the code's own disarm
        return FencesStop(_result(run))
    if not isinstance(e, asyncio.CancelledError):
        return e
    task = asyncio.current_task()
    if hasattr(task, "uncancel"):  # 3.11+: the caller's task isn't being cancelled, only this run
        task.uncancel()
    return FencesStop(_result(run))


def _outcome(run: RunState, exc: Optional[BaseException]):
    """(status, error) to report when the governed function exits."""
    if exc is None or isinstance(exc, GeneratorExit):  # GeneratorExit: consumer stopped a stream early
        return ("breached" if run.last_breach else "success"), None
    if isinstance(exc, FencesStop):  # a framework integration halted the agent on a breach
        return "breached", None
    if isinstance(exc, control.Halt):  # an outer run, halted after a stop, took this one with it
        run.last_breach = run.last_breach or "stopped_by_user"
        return "breached", None
    run.exception = events.exception_info(exc)  # includes cancellation
    e = run.exception
    return "error", f"{e['type']}: {e['message']}" if e["message"] else e["type"]


_noticed: set = set()  # (agent, code) pairs already logged by this process


def _notice(agent_name: str, code: str, message: str):
    """Log a server or SDK notice about an agent once per process, e.g. that it has no budget."""
    if (agent_name, code) not in _noticed:
        _noticed.add((agent_name, code))
        log.warning("agentfences: %s", message)


def _start_run(agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens, on_breach=None) -> RunState:
    run_id = str(uuid.uuid4())
    in_code = {"max_iterations": max_iterations, "max_duration_ms": max_duration_ms, "max_tokens": max_tokens}
    run = RunState(
        run_id=run_id,
        agent_name=agent_name,
        budget_usd=budget_usd,
        **{k: DEFAULT_LIMITS[k] if v is None else v for k, v in in_code.items()},
        context=events.current_context(),
        on_breach=on_breach or _on_breach,
        budget_in_code=budget_usd is not None,
    )
    client = _client
    if not client and budget_usd is None:
        _notice(agent_name, "no_budget", f"{agent_name} has no budget, so its spend isn't capped. "
                                         f"Set one with @governed(budget_usd=...).")
    if client:
        # The run must start even if redaction drops its context: limits depend on it
        sent = events.redact({"type": "run_start", "agent_name": agent_name, "context": dict(run.context)}) if run.context else None
        # Limits left out in code go as null: the server fills them from the agent's dashboard settings
        resp = client.start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens,
                                context=(sent or {}).get("context"))
        if resp.get("quarantined"):  # refused before any agent code runs
            raise AgentQuarantined(agent_name, resp.get("detail"))
        # The server caps limits at the agent's ceilings: enforce the same numbers if it becomes unreachable later
        # A null budget (unlimited) is taken only for a run whose code set none: a reply can't lift a budget from the code
        control.adopt(run, resp.get("limits") or {}, control.LIMIT_FIELDS, null_budget=budget_usd is None)
        for n in resp.get("notices") or ():
            _notice(agent_name, n.get("code", ""), n.get("message", "") + (f" {n['url']}" if n.get("url") else ""))
        if "limits" not in resp and run.budget_usd is None and "network_error" not in resp:
            _notice(agent_name, "no_budget", f"{agent_name} has no budget, so its spend isn't capped. "
                                             f"Set one on its page in the Fences dashboard.")
        if "network_error" in resp:
            _warn_unreachable(run, resp["network_error"])
    control.register(run)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    control.unregister(run)
    client = _client
    if client:
        # The run must end even if redaction drops its error details
        sent = events.redact({"type": "run_end", "status": status, "error": error, "exception": run.exception}) if error else None
        client.end_run(run.run_id, status=status, error=(sent or {}).get("error"), exception=(sent or {}).get("exception"),
                       breach=run.last_breach if status == "breached" else None)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _local_breach(run: RunState) -> Optional[str]:
    # Limits trip only once exceeded, same rule as the backend.
    if run.budget_usd is not None and round(run.cost_usd, 9) > run.budget_usd:  # round away float drift (0.02*5 != 0.10)
        return "budget_exceeded"
    if run.iterations > run.max_iterations:
        return "iteration_limit"
    if run.duration_ms > run.max_duration_ms:
        return "time_limit"
    if run.max_tokens > 0 and run.tokens_used > run.max_tokens:
        return "token_limit"
    return None


def _server_check(run: RunState, cost_delta_usd: float, tokens_delta: int) -> dict:
    # A step_id per step: a retried checkpoint is counted once on servers that support it
    return _client.checkpoint(run.run_id, cost_delta_usd, run.iterations, run.duration_ms, tokens_delta, step_id=uuid.uuid4().hex)


def _decide(run: RunState, resp: Optional[dict]) -> CheckpointResult:
    """
    Cloud mode: the backend decides. It sees spend from other processes and limits
    changed from the dashboard, so a raised limit lets the next checkpoint pass.
    Local rules apply in local_only mode, or when the backend can't be reached.
    """
    if resp is None or "network_error" in resp:
        if resp is not None:
            ending = _fail_closed or bool(resp.get("auth_rejected"))  # a key revoked mid-run: the run can't go on unchecked
            _warn_unreachable(run, resp["network_error"], ending)
            if ending:
                run.last_breach = "fences_unreachable"
                return _result(run)
        run.last_breach = _local_breach(run)
    else:
        control.apply(run, resp.get("commands"))
        # Backend is authoritative: adopt its totals and (on a breach) its current limits
        control.adopt(run, resp, control.TOTAL_FIELDS)
        control.adopt(run, resp, control.LIMIT_FIELDS, null_budget=not run.budget_in_code)
        if resp.get("ok"):
            run.last_breach = None
        else:  # a fresh breach, or a 409 because the run is still fenced from an earlier one
            run.last_breach = resp.get("breach") or run.last_breach or "limit_reached"
    return _result(run)


def _result(run: RunState) -> CheckpointResult:
    if run.stopped:  # a stop from the dashboard sticks, whatever a later response says
        run.last_breach = "stopped_by_user"
        run.stop_seen = True
    elif run.pause_expired:
        run.last_breach = "paused"
    return _breach(run) if run.last_breach else CheckpointResult(spent_usd=run.cost_usd)


def _breach(run: RunState) -> CheckpointResult:
    result = breach_result(
        run.last_breach,
        spent=run.cost_usd,
        limit=run.budget_usd or 0,  # None (no budget) can't be formatted; only budget breaches show it
        iterations=run.iterations,
        tokens_used=run.tokens_used,
        duration_ms=run.duration_ms,
    )
    result.spent_usd = run.cost_usd
    return result


def _holds_at(run: RunState, resp: Optional[dict], result: CheckpointResult) -> bool:
    """Wait at this breach? Only one the server decided: an unreachable server can't have its limit raised."""
    return (result.breach_type in control.HOLDABLE and control.holds(run)
            and resp is not None and "network_error" not in resp)


def _after_hold(run: RunState, breach: CheckpointResult) -> CheckpointResult:
    outcome = control.end_hold(run)
    if outcome == "released":  # an operator raised the limit and resumed the run
        run.last_breach = None
        return CheckpointResult(spent_usd=run.cost_usd)
    return _result(run) if outcome == "stopped" else breach  # timed out: the limit it was held at


def _record_step(cost_delta_usd: float, tokens_used: int) -> Optional[RunState]:
    run = get_active_run()
    if run is not None:
        with reserve_lock:  # the actual cost replaces the holds of calls that finished, in one step
            run.cost_usd += cost_delta_usd
            done = [h for h in run.holds if h[1]]
            run.holds[:] = [h for h in run.holds if not h[1]]
            run.reserved_usd -= sum(h[0] for h in done)
        run.tokens_used += tokens_used
        run.iterations += 1
    return run


def check_amount(value, what: str) -> float:
    """
    A cost or count as a float: a finite number, zero or more (an int, float or Decimal). A NaN
    or negative one would switch the budget off, so it raises instead.
    """
    if isinstance(value, (str, bytes, bool)) or not hasattr(value, "__float__"):
        raise TypeError(f"{what} must be a number, not {value!r}")
    number = float(value)
    if number != number:
        raise TypeError(f"{what} must be a number, not {value!r}")
    if not 0 <= number < float("inf"):
        raise ValueError(f"{what} must be zero or more and finite, not {value!r}")
    return number


def _measure(response, cost_delta_usd: float, tokens_used: int):
    """(cost, tokens) for one step: the response's priced usage plus any extra amounts passed."""
    if isinstance(response, (int, float)):  # checkpoint(0.02) would otherwise price 0.02 as a "response"
        raise TypeError("checkpoint() takes the model's response first; pass a cost as checkpoint(cost_delta_usd=...)")
    cost_delta_usd = check_amount(cost_delta_usd, "cost_delta_usd")
    check_amount(tokens_used, "tokens_used")
    if response is None:
        return cost_delta_usd, tokens_used
    cost, tokens = cost_of(response)
    return cost + cost_delta_usd, tokens + tokens_used


async def checkpoint(response=None, cost_delta_usd: float = 0.0, tokens_used: int = 0) -> CheckpointResult:
    """
    Record one step's spend and tokens, then check all governance limits.

    Call it after each model call with the model's response; Fences reads the model
    and token usage from it and works out the cost:

        resp = client.chat.completions.create(...)
        result = await checkpoint(resp)

    Returns a CheckpointResult — check result.breached before continuing.
    The backend call runs in a worker thread, so it never blocks the event loop.

    Args:
        response: The model's response (OpenAI, Anthropic, Gemini, LangChain, or any
            object or dict with the same usage fields). Optional.
        cost_delta_usd: Extra spend for this step in USD, e.g. a paid tool call.
            Without a response, the step's whole cost.
        tokens_used: Extra tokens for this step, added to the response's.

    Returns:
        CheckpointResult with:
            .ok            — True if within all limits
            .breached      — True if a limit was crossed
            .breach_type   — which limit was crossed
            .message       — human-readable, ready to return to user
            .system_prompt — inject into LLM context for graceful summarisation
            .spent_usd     — the run's total spend so far
    """
    if get_active_run() is None:
        return CheckpointResult()
    cost, tokens = _measure(response, cost_delta_usd, tokens_used)
    run = _record_step(cost, tokens)
    resp = await asyncio.to_thread(_server_check, run, cost, tokens) if _client else None
    result = _decide(run, resp)
    if _holds_at(run, resp, result):  # on_breach="hold": wait here for someone to raise the limit
        control.start_hold(run, "breach")
        await control.wait_if_paused_async(run)
        return _after_hold(run, result)
    if run.paused and not result.breached:  # paused from the dashboard: wait here until resumed
        await control.wait_if_paused_async(run)
        result = _result(run)
    return result


def checkpoint_sync(response=None, cost_delta_usd: float = 0.0, tokens_used: int = 0) -> CheckpointResult:
    """checkpoint() for synchronous agents. Same arguments and result."""
    if get_active_run() is None:
        return CheckpointResult()
    cost, tokens = _measure(response, cost_delta_usd, tokens_used)
    run = _record_step(cost, tokens)
    resp = _server_check(run, cost, tokens) if _client else None
    result = _decide(run, resp)
    if _holds_at(run, resp, result):
        control.start_hold(run, "breach")
        control.wait_if_paused(run)
        return _after_hold(run, result)
    if run.paused and not result.breached:
        control.wait_if_paused(run)
        result = _result(run)
    return result


# ── Decision trail and spans ──────────────────────────────────────────────────

def flush(timeout: float = 5.0):
    """Wait up to `timeout` seconds for queued decisions and events to reach the backend, and for
    on_stop/on_quarantine hooks to finish. Runs automatically at exit; call it yourself in serverless handlers."""
    deadline = time.time() + timeout
    outbox.wait(deadline)
    control.drain(max(0.0, deadline - time.time()))  # containment hooks, and command acknowledgements


def record_span(name: str, elapsed_s: float, attributes: dict, error: Optional[BaseException] = None):
    """
    Add a tool or model call to the active run's evidence as an OpenTelemetry-shaped span
    (GenAI semantic conventions): kept on the run, and sent to the backend through the
    redact hook in cloud mode. Does nothing outside a run.
    """
    run = get_active_run()
    if run is None:
        return
    end = time.time()
    attrs = {k: v for k, v in attributes.items() if v is not None}
    attrs["fences.iteration"] = run.iterations
    if error is not None:
        attrs["error.type"] = type(error).__qualname__
        attrs["fences.error.message"] = events.error_text(error)
        if isinstance(getattr(error, "status_code", None), int):  # e.g. 429 rate limit, 529 overloaded
            attrs["http.response.status_code"] = error.status_code
    span = {"name": name, "start_ts": end - elapsed_s, "end_ts": end, "status": "error" if error else "ok", "attributes": attrs}
    run.events.append(span)
    client = _client
    if client is None or not outbox.spans_supported:
        return
    sent = events.redact({"type": "span", **span})
    if sent:
        sent.pop("type", None)
        outbox.put_span(client, run.run_id, sent)


atexit.register(flush)


def log_decision(reasoning: str, action: Optional[str] = None):
    """
    Record the agent's reasoning at this step.

    Args:
        reasoning: Why the agent is taking this action.
        action: Short label for the action (e.g. "web_search").
    """
    run = get_active_run()
    if run is None:
        return

    entry = {
        "timestamp": time.time(),
        "iteration": run.iterations,
        "reasoning": reasoning,
        "action": action,
    }
    run.decisions.append(entry)

    client = _client
    sent = events.redact({"type": "decision", "reasoning": reasoning, "action": action}) if client else None
    if sent and sent.get("reasoning"):
        outbox.put_decision(client, {"run_id": run.run_id, "iteration": run.iterations,
                                     "reasoning": sent["reasoning"], "action": sent.get("action")})
