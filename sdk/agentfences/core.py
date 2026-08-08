import time
import uuid
import asyncio
import functools
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

from .exceptions import BudgetExceeded, IterationLimitReached, TimeLimitReached, TokenLimitReached
from .client import GovClient

_local = threading.local()


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

@dataclass
class RunState:
    run_id: str
    agent_name: str
    budget_usd: float
    max_iterations: int
    max_duration_ms: int
    max_tokens: int
    raise_on_breach: bool
    cost_usd: float = 0.0
    iterations: int = 0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.time)
    decisions: list = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


def get_active_run() -> Optional[RunState]:
    return getattr(_local, "run", None)


def _set_active_run(run: Optional[RunState]):
    _local.run = run


_client: Optional[GovClient] = None
_local_only: bool = False


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:8000",
    local_only: bool = False,
):
    """
    Initialize Fences. Call once at startup before using @governed.

    Local mode — no backend required:
        agentfences.init(local_only=True)

    Cloud mode — connects to a Fences backend:
        agentfences.init(api_key="fc_...", endpoint="https://...")
    """
    global _client, _local_only
    _local_only = local_only

    if local_only:
        _client = None
        return

    if not api_key:
        raise ValueError(
            "api_key is required unless local_only=True. "
            "Use agentfences.init(local_only=True) for local usage."
        )
    _client = GovClient(api_key=api_key, endpoint=endpoint)


def _get_client() -> Optional[GovClient]:
    return _client


def _require_init():
    if not _local_only and _client is None:
        raise RuntimeError(
            "Call agentfences.init() before using Fences. "
            "For local usage: agentfences.init(local_only=True)"
        )


def governed(
    budget_usd: float,
    max_iterations: int = 100,
    max_duration_ms: int = 300_000,
    max_tokens: int = 0,
    raise_on_breach: bool = False,
):
    """
    Decorator that applies governance policy to an agent function.

    Args:
        budget_usd: Maximum spend allowed for this run in USD.
        max_iterations: Maximum number of checkpoint() calls allowed.
        max_duration_ms: Maximum wall-clock duration in milliseconds.
        max_tokens: Maximum total tokens (input + output) allowed. 0 = no limit.
        raise_on_breach: If True, raises an exception on breach (legacy behaviour).
                         If False (default), checkpoint() returns a CheckpointResult
                         with breached=True so the agent can handle it gracefully.

    Usage:
        agentfences.init(local_only=True)

        @governed(budget_usd=0.50, max_iterations=20)
        async def run_agent(query: str):
            response = call_llm(query)
            result = await checkpoint(cost_delta_usd=0.02, tokens_used=response.usage.total_tokens)
            if result.breached:
                messages.append({"role": "system", "content": result.system_prompt})
                final = call_llm_summarize(messages)
                return final
            return response
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _require_init()
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens, raise_on_breach)
            _set_active_run(run)
            try:
                result = await func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except (BudgetExceeded, IterationLimitReached, TimeLimitReached, TokenLimitReached):
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _require_init()
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens, raise_on_breach)
            _set_active_run(run)
            try:
                result = func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except (BudgetExceeded, IterationLimitReached, TimeLimitReached, TokenLimitReached):
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _start_run(agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens, raise_on_breach) -> RunState:
    run_id = str(uuid.uuid4())
    run = RunState(
        run_id=run_id,
        agent_name=agent_name,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        max_duration_ms=max_duration_ms,
        max_tokens=max_tokens,
        raise_on_breach=raise_on_breach,
    )
    client = _get_client()
    if client:
        client.start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    client = _get_client()
    if client:
        client.end_run(run.run_id, status=status, error=error)


async def checkpoint(cost_delta_usd: float = 0.0, tokens_used: int = 0) -> CheckpointResult:
    """
    Report spend and token usage, then check all governance limits.

    Call this after each LLM call in your agent loop. Returns a
    CheckpointResult — check result.breached before continuing.

    Args:
        cost_delta_usd: Amount spent since the last checkpoint call.
        tokens_used: Total tokens (input + output) consumed by this step.

    Returns:
        CheckpointResult with:
            .ok            — True if within all limits
            .breached      — True if a limit was crossed
            .breach_type   — which limit was crossed
            .message       — human-readable, ready to return to user
            .system_prompt — inject into LLM context for graceful summarisation
    """
    run = get_active_run()
    if run is None:
        return CheckpointResult()

    run.cost_usd    += cost_delta_usd
    run.tokens_used += tokens_used
    run.iterations  += 1

    breach_result: Optional[CheckpointResult] = None

    if run.cost_usd >= run.budget_usd:
        breach_result = _make_breach_result(
            "budget_exceeded",
            spent=run.cost_usd,
            limit=run.budget_usd,
        )
    elif run.iterations >= run.max_iterations:
        breach_result = _make_breach_result(
            "iteration_limit",
            iterations=run.iterations,
        )
    elif run.duration_ms >= run.max_duration_ms:
        breach_result = _make_breach_result(
            "time_limit",
            duration_ms=run.duration_ms,
        )
    elif run.max_tokens > 0 and run.tokens_used >= run.max_tokens:
        breach_result = _make_breach_result(
            "token_limit",
            tokens_used=run.tokens_used,
        )

    if breach_result:
        _end_run(run, status="breached")
        if run.raise_on_breach:
            if breach_result.breach_type == "budget_exceeded":
                raise BudgetExceeded(run.cost_usd, run.budget_usd)
            elif breach_result.breach_type == "iteration_limit":
                raise IterationLimitReached(run.iterations, run.max_iterations)
            elif breach_result.breach_type == "time_limit":
                raise TimeLimitReached(run.duration_ms, run.max_duration_ms)
            elif breach_result.breach_type == "token_limit":
                raise TokenLimitReached(run.tokens_used, run.max_tokens)
        return breach_result

    # Server check — cloud mode only
    client = _get_client()
    if client is None:
        return CheckpointResult()

    result = client.checkpoint(
        run.run_id, cost_delta_usd, run.iterations, run.duration_ms, run.tokens_used
    )

    if result.get("ok", True):
        return CheckpointResult()

    breach = result.get("breach")
    server_result = _make_breach_result(
        breach,
        spent=result.get("spent_usd", run.cost_usd),
        limit=result.get("budget_usd", run.budget_usd),
        iterations=result.get("iterations", run.iterations),
        tokens_used=result.get("tokens_used", run.tokens_used),
        duration_ms=run.duration_ms,
    )
    _end_run(run, status="breached")
    if run.raise_on_breach:
        if breach == "budget_exceeded":
            raise BudgetExceeded(run.cost_usd, run.budget_usd)
        elif breach == "iteration_limit":
            raise IterationLimitReached(run.iterations, run.max_iterations)
        elif breach == "time_limit":
            raise TimeLimitReached(run.duration_ms, run.max_duration_ms)
        elif breach == "token_limit":
            raise TokenLimitReached(run.tokens_used, run.max_tokens)
    return server_result


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

    client = _get_client()
    if client:
        client.log_decision(
            run_id=run.run_id,
            iteration=run.iterations,
            reasoning=reasoning,
            action=action,
        )