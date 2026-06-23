import time
import uuid
import asyncio
import functools
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

from .exceptions import BudgetExceeded, IterationLimitReached, TimeLimitReached
from .client import GovClient

_local = threading.local()


@dataclass
class RunState:
    run_id: str
    agent_name: str
    budget_usd: float
    max_iterations: int
    max_duration_ms: int
    cost_usd: float = 0.0
    iterations: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


def get_active_run() -> Optional[RunState]:
    return getattr(_local, "run", None)


def _set_active_run(run: Optional[RunState]):
    _local.run = run


_client: Optional[GovClient] = None


def init(api_key: str, endpoint: str = "http://localhost:8000"):
    """Initialize Fences. Call once at startup, before using @governed."""
    global _client
    _client = GovClient(api_key=api_key, endpoint=endpoint)


def _require_client() -> GovClient:
    if _client is None:
        raise RuntimeError("Call fences.init(api_key=...) before using Fences")
    return _client


def governed(
    budget_usd: float,
    max_iterations: int = 100,
    max_duration_ms: int = 300_000,  # 5 minutes default
):
    """
    Decorate an agent entrypoint with governance policy.

    Usage:
        @fences.governed(budget_usd=0.50, max_iterations=20, max_duration_ms=60000)
        async def run_agent(query: str):
            ...
            await fences.checkpoint(cost_delta_usd=0.02)
            ...
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms)
            _set_active_run(run)
            try:
                result = await func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except (BudgetExceeded, IterationLimitReached, TimeLimitReached):
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            run = _start_run(func.__name__, budget_usd, max_iterations, max_duration_ms)
            _set_active_run(run)
            try:
                result = func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except (BudgetExceeded, IterationLimitReached, TimeLimitReached):
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _start_run(
    agent_name: str,
    budget_usd: float,
    max_iterations: int,
    max_duration_ms: int,
) -> RunState:
    run_id = str(uuid.uuid4())
    run = RunState(
        run_id=run_id,
        agent_name=agent_name,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        max_duration_ms=max_duration_ms,
    )
    _require_client().start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    _require_client().end_run(run.run_id, status=status, error=error)


async def checkpoint(cost_delta_usd: float = 0.0):
    """
    Call at each decision point in your agent loop. Enforces all three
    limits — budget, iterations, and time — locally first (instant),
    then confirms with the server (source of truth).

    Raises BudgetExceeded, IterationLimitReached, or TimeLimitReached.
    """
    run = get_active_run()
    if run is None:
        return  # Not inside a @governed function — no-op

    run.cost_usd += cost_delta_usd
    run.iterations += 1

    # ── 1. Local checks — instant, no network ────────────────────────────────
    if run.cost_usd >= run.budget_usd:
        raise BudgetExceeded(run.cost_usd, run.budget_usd)

    if run.iterations >= run.max_iterations:
        raise IterationLimitReached(run.iterations, run.max_iterations)

    if run.duration_ms >= run.max_duration_ms:
        raise TimeLimitReached(run.duration_ms, run.max_duration_ms)

    # ── 2. Server check — authoritative source of truth ─────────────────────
    result = _require_client().checkpoint(
        run.run_id, cost_delta_usd, run.iterations, run.duration_ms
    )

    if result.get("ok", True):
        return

    # Server says we're over — figure out which limit was breached
    breach = result.get("breach")
    if breach == "budget_exceeded":
        raise BudgetExceeded(result.get("spent_usd", run.cost_usd), result.get("budget_usd", run.budget_usd))
    elif breach == "iteration_limit":
        raise IterationLimitReached(run.iterations, run.max_iterations)
    elif breach == "time_limit":
        raise TimeLimitReached(run.duration_ms, run.max_duration_ms)