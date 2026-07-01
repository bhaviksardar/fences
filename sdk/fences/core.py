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
_local_only: bool = False


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:8000",
    local_only: bool = False,
):
    """
    Initialize Fences.

    Free / self-hosted usage — no backend, no API key needed:
        fences.init(local_only=True)

    Fences Cloud (managed backend, audit trail, dashboard):
        fences.init(api_key="fc_...", endpoint="https://your-fences-instance.com")

    local_only mode enforces all three limits (budget, iterations, time)
    entirely in-process. No data is sent anywhere. The tradeoff is that
    enforcement isn't server-authoritative — a restarted process starts
    fresh — but for most self-hosted use cases that's fine.
    """
    global _client, _local_only
    _local_only = local_only

    if local_only:
        _client = None
        return

    if not api_key:
        raise ValueError(
            "api_key is required unless local_only=True. "
            "Use fences.init(local_only=True) for free self-hosted usage, "
            "or fences.init(api_key='fc_...') for Fences Cloud."
        )
    _client = GovClient(api_key=api_key, endpoint=endpoint)


def _get_client() -> Optional[GovClient]:
    """Returns the client, or None in local_only mode. Never raises."""
    return _client


def _require_init():
    """Raises if fences.init() hasn't been called at all."""
    if not _local_only and _client is None:
        raise RuntimeError(
            "Call fences.init() before using Fences. "
            "For free local usage: fences.init(local_only=True)"
        )


def governed(
    budget_usd: float,
    max_iterations: int = 100,
    max_duration_ms: int = 300_000,
):
    """
    Decorate an agent entrypoint with governance policy.

    Works in both local_only and cloud modes — no code change needed
    when upgrading from free to paid.

    Usage:
        fences.init(local_only=True)  # free tier

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
            _require_init()
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
            _require_init()
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
    client = _get_client()
    if client:
        client.start_run(run_id, agent_name, budget_usd, max_iterations, max_duration_ms)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    client = _get_client()
    if client:
        client.end_run(run.run_id, status=status, error=error)


async def checkpoint(cost_delta_usd: float = 0.0):
    """
    Call at each decision point in your agent loop.

    In local_only mode: enforces limits purely in-process, instant,
    no network call.

    In cloud mode: enforces locally first (fast), then confirms with
    the server (authoritative — catches spend from other processes or
    manually-adjusted budgets).

    Raises BudgetExceeded, IterationLimitReached, or TimeLimitReached.
    """
    run = get_active_run()
    if run is None:
        return

    run.cost_usd += cost_delta_usd
    run.iterations += 1

    # ── Local checks — always run, instant ───────────────────────────────────
    if run.cost_usd >= run.budget_usd:
        raise BudgetExceeded(run.cost_usd, run.budget_usd)

    if run.iterations >= run.max_iterations:
        raise IterationLimitReached(run.iterations, run.max_iterations)

    if run.duration_ms >= run.max_duration_ms:
        raise TimeLimitReached(run.duration_ms, run.max_duration_ms)

    # ── Server check — cloud mode only ───────────────────────────────────────
    client = _get_client()
    if client is None:
        return  # local_only mode — local checks above are the full enforcement

    result = client.checkpoint(
        run.run_id, cost_delta_usd, run.iterations, run.duration_ms
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


def log_decision(reasoning: str, action: Optional[str] = None):
    """
    Record WHY the agent is doing something at this step.

    In local_only mode: stored in-memory on the RunState only (not
    persisted anywhere, since there's no backend). Useful for debugging
    locally — call fences.get_active_run().decisions to inspect.

    In cloud mode: sent to the backend and queryable via the dashboard.
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

    # Always store in-memory regardless of mode
    if not hasattr(run, "decisions"):
        run.decisions = []
    run.decisions.append(entry)

    # Send to backend in cloud mode only
    client = _get_client()
    if client:
        client.log_decision(
            run_id=run.run_id,
            iteration=run.iterations,
            reasoning=reasoning,
            action=action,
        )