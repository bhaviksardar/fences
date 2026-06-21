import time
import uuid
import asyncio
import functools
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

from .exceptions import BudgetExceeded
from .client import GovClient

_local = threading.local()


@dataclass
class RunState:
    run_id: str
    agent_name: str
    budget_usd: float
    cost_usd: float = 0.0


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


def governed(budget_usd: float):
    """
    Decorate an agent entrypoint with a budget limit.

    Usage:
        @fences.governed(budget_usd=0.50)
        async def run_agent(query: str):
            ...
            await fences.checkpoint(cost_delta_usd=0.02)
            ...
    """
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            run = _start_run(func.__name__, budget_usd)
            _set_active_run(run)
            try:
                result = await func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except BudgetExceeded:
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            run = _start_run(func.__name__, budget_usd)
            _set_active_run(run)
            try:
                result = func(*args, **kwargs)
                _end_run(run, status="success")
                return result
            except BudgetExceeded:
                _end_run(run, status="breached")
                raise
            except Exception as e:
                _end_run(run, status="error", error=str(e))
                raise
            finally:
                _set_active_run(None)

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _start_run(agent_name: str, budget_usd: float) -> RunState:
    run_id = str(uuid.uuid4())
    run = RunState(run_id=run_id, agent_name=agent_name, budget_usd=budget_usd)
    _require_client().start_run(run_id, agent_name, budget_usd)
    return run


def _end_run(run: RunState, status: str, error: Optional[str] = None):
    _require_client().end_run(run.run_id, status=status, error=error)


async def checkpoint(cost_delta_usd: float = 0.0):
    """
    Report spend since the last checkpoint and check the budget.

    Checks LOCALLY first (instant) — if the local tally already shows
    we're over budget, raise immediately without waiting on the network.
    Otherwise, confirm with the server (the source of truth) before
    continuing, since the server may know about spend this process isn't
    aware of (e.g. a parallel run, or a manually-adjusted budget).

    Raises BudgetExceeded if the run is over budget, by either account.
    """
    run = get_active_run()
    if run is None:
        return  # Not inside a @governed function — no-op, don't break caller's code

    run.cost_usd += cost_delta_usd

    # 1. Local check — instant, no network wait
    if run.cost_usd >= run.budget_usd:
        raise BudgetExceeded(run.cost_usd, run.budget_usd)

    # 2. Server check — authoritative, source of truth
    result = _require_client().checkpoint(run.run_id, cost_delta_usd)
    if not result.get("ok", True):
        server_spent = result.get("spent_usd", run.cost_usd)
        server_budget = result.get("budget_usd", run.budget_usd)
        raise BudgetExceeded(server_spent, server_budget)