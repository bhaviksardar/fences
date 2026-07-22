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
        fences.init(local_only=True)

    Cloud mode — connects to a Fences backend:
        fences.init(api_key="fc_...", endpoint="https://...")
    """
    global _client, _local_only
    _local_only = local_only

    if local_only:
        _client = None
        return

    if not api_key:
        raise ValueError(
            "api_key is required unless local_only=True. "
            "Use fences.init(local_only=True) for local usage."
        )
    _client = GovClient(api_key=api_key, endpoint=endpoint)


def _get_client() -> Optional[GovClient]:
    return _client


def _require_init():
    if not _local_only and _client is None:
        raise RuntimeError(
            "Call fences.init() before using Fences. "
            "For local usage: fences.init(local_only=True)"
        )


def governed(
    budget_usd: float,
    max_iterations: int = 100,
    max_duration_ms: int = 300_000,
    max_tokens: int = 0,
):
    """
    Decorator that applies governance policy to an agent function.

    Args:
        budget_usd: Maximum spend allowed for this run in USD.
        max_iterations: Maximum number of checkpoint() calls allowed.
        max_duration_ms: Maximum wall-clock duration in milliseconds.
        max_tokens: Maximum total tokens (input + output) allowed. 0 = no limit.

    Raises BudgetExceeded, IterationLimitReached, TimeLimitReached, or
    TokenLimitReached when limits are crossed.

    Usage:
        @fences.governed(budget_usd=0.50, max_iterations=20, max_tokens=50000)
        async def run_agent(query: str):
            response = call_llm(query)
            await fences.checkpoint(
                cost_delta_usd=0.02,
                tokens_used=response.usage.total_tokens,
            )
            return response
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _require_init()
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens)
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
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms, max_tokens)
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


def _start_run(agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens) -> RunState:
    run_id = str(uuid.uuid4())
    run = RunState(
        run_id=run_id,
        agent_name=agent_name,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        max_duration_ms=max_duration_ms,
        max_tokens=max_tokens,
    )
    client = _get_client()
    if client:
        client.start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms, max_tokens)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    client = _get_client()
    if client:
        client.end_run(run.run_id, status=status, error=error)


async def checkpoint(cost_delta_usd: float = 0.0, tokens_used: int = 0):
    """
    Report spend and token usage, then check all governance limits.

    Call this after each LLM call in your agent loop.
    Raises BudgetExceeded, IterationLimitReached, TimeLimitReached,
    or TokenLimitReached if any limit is crossed.

    Args:
        cost_delta_usd: Amount spent since the last checkpoint call.
        tokens_used: Total tokens (input + output) consumed by this step.
    """
    run = get_active_run()
    if run is None:
        return

    run.cost_usd    += cost_delta_usd
    run.tokens_used += tokens_used
    run.iterations  += 1

    # Local checks — instant, no network
    if run.cost_usd >= run.budget_usd:
        raise BudgetExceeded(run.cost_usd, run.budget_usd)

    if run.iterations >= run.max_iterations:
        raise IterationLimitReached(run.iterations, run.max_iterations)

    if run.duration_ms >= run.max_duration_ms:
        raise TimeLimitReached(run.duration_ms, run.max_duration_ms)

    if run.max_tokens > 0 and run.tokens_used >= run.max_tokens:
        raise TokenLimitReached(run.tokens_used, run.max_tokens)

    # Server check — cloud mode only
    client = _get_client()
    if client is None:
        return

    result = client.checkpoint(
        run.run_id, cost_delta_usd, run.iterations, run.duration_ms, run.tokens_used
    )

    if result.get("ok", True):
        return

    breach = result.get("breach")
    if breach == "budget_exceeded":
        raise BudgetExceeded(result.get("spent_usd", run.cost_usd), result.get("budget_usd", run.budget_usd))
    elif breach == "iteration_limit":
        raise IterationLimitReached(run.iterations, run.max_iterations)
    elif breach == "time_limit":
        raise TimeLimitReached(run.duration_ms, run.max_duration_ms)
    elif breach == "token_limit":
        raise TokenLimitReached(run.tokens_used, run.max_tokens)


def log_decision(reasoning: str, action: Optional[str] = None):
    """
    Record the agent's reasoning at this step.

    Call this before any significant action to build an audit trail
    of why the agent did what it did.

    Args:
        reasoning: Why the agent is taking this action.
        action: Short label for the action being taken (e.g. "web_search").
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