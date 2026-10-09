import time
import uuid
import queue
import atexit
import asyncio
import inspect
import logging
import functools
import threading
import contextvars
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable

from .exceptions import AgentQuarantined
from .client import GovClient
from .pricing import cost_of, set_custom_prices
from . import events, control

log = logging.getLogger("agentfences")

# The active run is per task/context, not per thread: concurrent asyncio agents
# share one thread, and nested @governed calls must restore the outer run.
_active = contextvars.ContextVar("agentfences_run", default=None)


# ── CheckpointResult ──────────────────────────────────────────────────────────

@dataclass
class CheckpointResult:
    """
    Returned by checkpoint() after every call.

    If breached is False, the agent is within all limits and should continue.
    If breached is True, the agent has crossed a limit and should stop or
    handle gracefully — the message field is ready to return to the user
    or inject into the LLM conversation as a system message.

    Usage — programmatic check:
        result = await checkpoint(cost_delta_usd=0.02, tokens_used=450)
        if result.breached:
            return result.message

    Usage — inject into LLM conversation:
        result = await checkpoint(cost_delta_usd=0.02, tokens_used=450)
        if result.breached:
            messages.append({"role": "system", "content": result.system_prompt})
            final = client.chat.completions.create(model=MODEL, messages=messages)
            return final.choices[0].message.content
    """
    breached: bool = False
    breach_type: Optional[str] = None   # "budget_exceeded" | "iteration_limit" | "time_limit" | "token_limit"
    message: str = ""                   # human-readable, ready to return to user
    system_prompt: str = ""             # inject into LLM conversation context
    spent_usd: float = 0.0              # the run's total spend so far

    @property
    def ok(self) -> bool:
        return not self.breached


def _make_breach_result(breach_type: str, **kwargs) -> CheckpointResult:
    messages = {
        "budget_exceeded": (
            f"I've reached my budget limit (spent ${kwargs.get('spent', 0):.4f} "
            f"of ${kwargs.get('limit', 0):.4f}). I'll summarize what I found so far."
        ),
        "iteration_limit": (
            f"I've reached my iteration limit ({kwargs.get('iterations', 0)} steps). "
            f"I'll summarize what I found so far."
        ),
        "time_limit": (
            f"I've reached my time limit ({kwargs.get('duration_ms', 0)}ms). "
            f"I'll summarize what I found so far."
        ),
        "token_limit": (
            f"I've reached my token limit ({kwargs.get('tokens_used', 0):,} tokens). "
            f"I'll summarize what I found so far."
        ),
        "stopped_by_user": "I was stopped from the Fences dashboard. I'll summarize what I found so far.",
        "key_daily_budget": "My daily spending cap has been reached. I'll summarize what I found so far.",
        "key_monthly_budget": "My monthly spending cap has been reached. I'll summarize what I found so far.",
        "fences_unreachable": (
            "I can't reach Fences to confirm I'm within my limits, so I'm stopping to be safe. "
            "I'll summarize what I found so far."
        ),
        "paused": "I was paused from the Fences dashboard and not resumed in time. I'll summarize what I found so far.",
    }

    system_prompts = {
        "budget_exceeded": (
            f"You have reached your budget limit (${kwargs.get('spent', 0):.4f} of "
            f"${kwargs.get('limit', 0):.4f} spent). Stop your current task immediately "
            f"and provide a clear summary of what you have found or completed so far. "
            f"Tell the user you stopped due to budget and what you accomplished."
        ),
        "iteration_limit": (
            f"You have reached your iteration limit ({kwargs.get('iterations', 0)} steps). "
            f"Stop your current task immediately and summarize what you have found or "
            f"completed so far. Tell the user you stopped due to the iteration limit."
        ),
        "time_limit": (
            f"You have reached your time limit. Stop your current task immediately "
            f"and summarize what you have found or completed so far. "
            f"Tell the user you stopped due to the time limit."
        ),
        "token_limit": (
            f"You have reached your token limit ({kwargs.get('tokens_used', 0):,} tokens used). "
            f"Stop your current task immediately and summarize what you have found or "
            f"completed so far. Tell the user you stopped due to the token limit."
        ),
        "stopped_by_user": (
            "A person stopped this run from the Fences dashboard. Stop your current task immediately "
            "and summarize what you have found or completed so far. Tell the user you were stopped by an operator."
        ),
        "key_daily_budget": (
            "Your daily spending cap, across all of your runs, has been reached. Stop your current task "
            "immediately and summarize what you have found or completed so far. Tell the user you stopped "
            "because the daily cap was reached."
        ),
        "key_monthly_budget": (
            "Your monthly spending cap, across all of your runs, has been reached. Stop your current task "
            "immediately and summarize what you have found or completed so far. Tell the user you stopped "
            "because the monthly cap was reached."
        ),
        "fences_unreachable": (
            "Your governance service can't be reached and you are configured to stop when that happens. "
            "Stop your current task immediately and summarize what you have found or completed so far."
        ),
        "paused": (
            "A person paused this run from the Fences dashboard and it was not resumed in time. Stop your current "
            "task and summarize what you have found or completed so far. Tell the user the run was paused by an operator."
        ),
    }

    msg = messages.get(breach_type, "Governance limit reached.")
    sys_prompt = system_prompts.get(breach_type, "A governance limit has been reached. Summarize what you have done so far.")

    return CheckpointResult(
        breached=True,
        breach_type=breach_type,
        message=msg,
        system_prompt=sys_prompt,
    )


# ── RunState ──────────────────────────────────────────────────────────────────

def _set_event() -> threading.Event:
    e = threading.Event()
    e.set()  # not paused
    return e


MAX_EVENTS_PER_RUN = 1000
EVENT_BATCH = 50


@dataclass
class RunState:
    run_id: str
    agent_name: str
    budget_usd: float
    max_iterations: int
    max_duration_ms: int
    max_tokens: int
    cost_usd: float = 0.0
    iterations: int = 0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.time)
    decisions: list = field(default_factory=list)
    last_breach: Optional[str] = None   # breach from the latest checkpoint; cleared once limits are raised
    warned: bool = False
    context: dict = field(default_factory=dict)  # environment, release and agentfences.context() values
    # Tool and model calls, newest kept. ponytail: capped per run so a runaway agent can't eat memory
    events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS_PER_RUN))
    exception: Optional[dict] = None             # type, message and stack if the run raised
    # Live control from the dashboard (see control.py)
    stopped: bool = False
    paused: bool = False
    pause_expired: bool = False
    resumed: threading.Event = field(default_factory=lambda: _set_event())
    approvals: dict = field(default_factory=dict)  # approval_id -> Approval, or None while waiting
    awaiting: int = 0

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


def get_active_run() -> Optional[RunState]:
    return _active.get()


def _set_active_run(run: Optional[RunState]):
    return _active.set(run)


def _reset_active_run(token):
    try:
        _active.reset(token)
    except ValueError:  # async generator finalized from a different context
        pass


_client: Optional[GovClient] = None
_local_only: bool = False
_fail_closed: bool = False


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
    run waits at its next checkpoint, for up to pause_timeout_s seconds.
    """
    global _client, _local_only, _fail_closed
    set_custom_prices(prices)
    events.configure(environment, release, redact)
    if instrument:
        from .instrument import instrument as patch_clients
        patch_clients()
    _local_only = local_only
    _fail_closed = fail_closed

    if local_only:
        _client = None
        control.configure(None, heartbeat_s, pause_timeout_s)
        return

    if not api_key:
        raise ValueError(
            "api_key is required unless local_only=True. "
            "Use agentfences.init(local_only=True) for local usage."
        )
    _client = GovClient(api_key=api_key, endpoint=endpoint)
    _start_decision_worker()
    control.configure(_client, heartbeat_s, pause_timeout_s)


def _require_init():
    if not _local_only and _client is None:
        raise RuntimeError(
            "Call agentfences.init() before using Fences. "
            "For local usage: agentfences.init(local_only=True)"
        )


def _warn_unreachable(run: RunState, err: str):
    if not run.warned:
        run.warned = True
        log.warning("Fences backend unreachable for run %s (%s); enforcing limits locally", run.run_id, err)


# ── Decorator ─────────────────────────────────────────────────────────────────

def governed(
    budget_usd: float,
    max_iterations: int = 100,
    max_duration_ms: int = 300_000,
    max_tokens: int = 0,
):
    """
    Decorator that applies governance policy to an agent function.
    Works on sync functions, async functions and async generators (streaming agents).

    Args:
        budget_usd: Maximum spend allowed for this run in USD.
        max_iterations: Maximum number of checkpoint() calls allowed.
        max_duration_ms: Maximum wall-clock duration in milliseconds.
        max_tokens: Maximum total tokens (input + output) allowed. 0 = no limit.

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
    def decorator(func: Callable) -> Callable:
        start = functools.partial(_start_run, func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens)

        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def agen_wrapper(*args, **kwargs):
                _require_init()
                run = await asyncio.to_thread(start)
                token, exc = _set_active_run(run), None
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                except BaseException as e:
                    exc = e
                    raise
                finally:
                    _reset_active_run(token)
                    await asyncio.to_thread(_end_run, run, *_outcome(run, exc))
            return agen_wrapper

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _require_init()
                run = await asyncio.to_thread(start)
                token, exc = _set_active_run(run), None
                try:
                    return await func(*args, **kwargs)
                except BaseException as e:
                    exc = e
                    raise
                finally:
                    _reset_active_run(token)
                    await asyncio.to_thread(_end_run, run, *_outcome(run, exc))
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _require_init()
            run = start()
            token, exc = _set_active_run(run), None
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                exc = e
                raise
            finally:
                _reset_active_run(token)
                _end_run(run, *_outcome(run, exc))
        return sync_wrapper

    return decorator


def _outcome(run: RunState, exc: Optional[BaseException]):
    """(status, error) to report when the governed function exits."""
    if exc is None or isinstance(exc, GeneratorExit):  # GeneratorExit: consumer stopped a stream early
        return ("breached" if run.last_breach else "success"), None
    run.exception = events.exception_info(exc)  # includes cancellation
    e = run.exception
    return "error", f"{e['type']}: {e['message']}" if e["message"] else e["type"]


def _start_run(agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens) -> RunState:
    run_id = str(uuid.uuid4())
    run = RunState(
        run_id=run_id,
        agent_name=agent_name,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        max_duration_ms=max_duration_ms,
        max_tokens=max_tokens,
        context=events.current_context(),
    )
    client = _client
    if client:
        # The run must start even if redaction drops its context: limits depend on it
        sent = events.redact({"type": "run_start", "agent_name": agent_name, "context": dict(run.context)}) if run.context else None
        resp = client.start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens,
                                context=(sent or {}).get("context"))
        if resp.get("quarantined"):  # refused before any agent code runs
            raise AgentQuarantined(agent_name, resp.get("detail"))
        # The server caps limits at the agent's ceilings: enforce the same numbers if it becomes unreachable later
        for k, v in (resp.get("limits") or {}).items():
            if k in control.LIMIT_FIELDS and isinstance(v, (int, float)):
                setattr(run, k, v)
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
        client.end_run(run.run_id, status=status, error=(sent or {}).get("error"), exception=(sent or {}).get("exception"))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _local_breach(run: RunState) -> Optional[str]:
    # Limits trip only once exceeded, same rule as the backend.
    if round(run.cost_usd, 9) > run.budget_usd:  # round away float drift (0.02*5 != 0.10)
        return "budget_exceeded"
    if run.iterations > run.max_iterations:
        return "iteration_limit"
    if run.duration_ms > run.max_duration_ms:
        return "time_limit"
    if run.max_tokens > 0 and run.tokens_used > run.max_tokens:
        return "token_limit"
    return None


def _server_check(run: RunState, cost_delta_usd: float, tokens_delta: int) -> dict:
    return _client.checkpoint(run.run_id, cost_delta_usd, run.iterations, run.duration_ms, tokens_delta)


def _decide(run: RunState, resp: Optional[dict]) -> CheckpointResult:
    """
    Cloud mode: the backend decides. It sees spend from other processes and limits
    changed from the dashboard, so a raised limit lets the next checkpoint pass.
    Local rules apply in local_only mode, or when the backend can't be reached.
    """
    if resp is None or "network_error" in resp:
        if resp is not None:
            _warn_unreachable(run, resp["network_error"])
            if _fail_closed:
                run.last_breach = "fences_unreachable"
                return _breach(run)
        run.last_breach = _local_breach(run)
    else:
        control.apply(run, resp.get("commands"))
        # Backend is authoritative: adopt its totals and (on a breach) its current limits
        run.cost_usd = resp.get("spent_usd", run.cost_usd)
        run.iterations = resp.get("iterations", run.iterations)
        run.tokens_used = resp.get("tokens_used", run.tokens_used)
        run.budget_usd = resp.get("budget_usd", run.budget_usd)
        run.max_iterations = resp.get("max_iterations", run.max_iterations)
        run.max_tokens = resp.get("max_tokens", run.max_tokens)
        run.max_duration_ms = resp.get("max_duration_ms", run.max_duration_ms)
        if resp.get("ok"):
            run.last_breach = None
        else:  # a fresh breach, or a 409 because the run is still fenced from an earlier one
            run.last_breach = resp.get("breach") or run.last_breach or "limit_reached"
    return _result(run)


def _result(run: RunState) -> CheckpointResult:
    if run.stopped:  # a stop from the dashboard sticks, whatever a later response says
        run.last_breach = "stopped_by_user"
    elif run.pause_expired:
        run.last_breach = "paused"
    return _breach(run) if run.last_breach else CheckpointResult(spent_usd=run.cost_usd)


def _breach(run: RunState) -> CheckpointResult:
    result = _make_breach_result(
        run.last_breach,
        spent=run.cost_usd,
        limit=run.budget_usd,
        iterations=run.iterations,
        tokens_used=run.tokens_used,
        duration_ms=run.duration_ms,
    )
    result.spent_usd = run.cost_usd
    return result


def _record_step(cost_delta_usd: float, tokens_used: int) -> Optional[RunState]:
    run = get_active_run()
    if run is not None:
        run.cost_usd += cost_delta_usd
        run.tokens_used += tokens_used
        run.iterations += 1
    return run


def _measure(response, cost_delta_usd: float, tokens_used: int):
    """(cost, tokens) for one step: the response's priced usage plus any extra amounts passed."""
    if isinstance(response, (int, float)):  # checkpoint(0.02) would otherwise price 0.02 as a "response"
        raise TypeError("checkpoint() takes the model's response first; pass a cost as checkpoint(cost_delta_usd=...)")
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
    result = _decide(run, _server_check(run, cost, tokens) if _client else None)
    if run.paused and not result.breached:
        control.wait_if_paused(run)
        result = _result(run)
    return result


# ── Decision trail ────────────────────────────────────────────────────────────

# Decisions and events go to the backend from one background thread, in order, so
# log_decision() and recorded calls never block the agent on a network round trip.
# Items: ("decision", client, payload) or ("event", client, run_id, event).
_decisions: "queue.Queue" = queue.Queue()
_worker: Optional[threading.Thread] = None
_events_supported = True  # off once a server says it has no events endpoint


def _send(item, more: list):
    """Send one item; consecutive events for the same run go in one request."""
    global _events_supported
    if item[0] == "decision":
        item[1].log_decision(**item[2])
        return
    _, client, run_id, event = item
    batch = [event]
    while more and more[0][0] == "event" and more[0][2] == run_id and len(batch) < EVENT_BATCH:
        batch.append(more.pop(0)[3])
    if _events_supported and client.log_events(run_id, batch).get("unsupported"):
        _events_supported = False
        log.warning("agentfences: this Fences server doesn't accept tool and model call events yet; "
                    "they're kept on the run locally but not sent")


def _decision_worker():
    pending: list = []
    while True:
        if not pending:
            pending.append(_decisions.get())
        while True:  # take everything queued so far, so a burst of events goes out together
            try:
                pending.append(_decisions.get_nowait())
            except queue.Empty:
                break
        item = pending.pop(0)
        before = len(pending)
        try:
            _send(item, pending)
        except Exception:
            pass
        finally:
            for _ in range(1 + before - len(pending)):  # this item plus any batched with it
                _decisions.task_done()


def _start_decision_worker():
    global _worker
    if _worker is None:
        _worker = threading.Thread(target=_decision_worker, name="agentfences-decisions", daemon=True)
        _worker.start()


def flush(timeout: float = 5.0):
    """Wait up to `timeout` seconds for queued decisions and events to reach the backend.
    Runs automatically at exit; call it yourself in serverless handlers."""
    deadline = time.time() + timeout
    while _decisions.unfinished_tasks and time.time() < deadline:
        time.sleep(0.02)


def record_event(event: dict):
    """
    Add a tool or model call to the active run's evidence: kept on the run, and sent to
    the backend (through the redact hook) in cloud mode. Does nothing outside a run.
    """
    run = get_active_run()
    if run is None:
        return
    event = {"ts": time.time(), "iteration": run.iterations, **event}
    run.events.append(event)
    client = _client
    if client is None or not _events_supported:
        return
    sent = events.redact(event)
    if sent:
        _decisions.put(("event", client, run.run_id, sent))


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
        _decisions.put(("decision", client, {"run_id": run.run_id, "iteration": run.iterations,
                                             "reasoning": sent["reasoning"], "action": sent.get("action")}))
