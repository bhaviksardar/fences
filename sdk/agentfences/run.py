"""A run's state, and which run is active in this task or thread."""
import asyncio
import contextvars
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# The active run is per task/context, not per thread: concurrent asyncio agents
# share one thread, and nested @governed calls must restore the outer run.
_active = contextvars.ContextVar("agentfences_run", default=None)


# ── RunState ──────────────────────────────────────────────────────────────────

def _set_event() -> threading.Event:
    e = threading.Event()
    e.set()  # not paused
    return e


MAX_EVENTS_PER_RUN = 1000


@dataclass
class RunState:
    run_id: str
    agent_name: str
    budget_usd: Optional[float]  # None: no budget anywhere, unlimited spend
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
    stopped_at: Optional[float] = None  # monotonic time a stop arrived
    stop_seen: bool = False             # the agent got the stop from a checkpoint (or a blocked tool)
    task: Optional[asyncio.Task] = None  # an async run's task, to cancel if a stop goes unseen
    thread: Optional[int] = None         # a sync run's thread while its code runs, to halt if a stop goes unseen
    loop: Optional[asyncio.AbstractEventLoop] = None
    cancelled: bool = False             # Fences cancelled the task (or halted the thread) after a stop
    reserved_usd: float = 0.0           # held by guarded tools and model calls in flight
    holds: list = field(default_factory=list)  # model calls' worst-case costs: [amount, finished]
    quarantined: bool = False           # the agent was quarantined from the dashboard while this ran
    on_breach: Optional[str] = None     # "hold": wait at a limit for someone to raise it
    held: Optional[str] = None          # while waiting at a limit: "breach" or "budget"
    acked: set = field(default_factory=set)  # command ids already acknowledged

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


# Checking room in a budget and reserving it is one step, across threads and tasks
reserve_lock = threading.Lock()


def get_active_run() -> Optional[RunState]:
    return _active.get()


def _set_active_run(run: Optional[RunState]):
    return _active.set(run)


def _reset_active_run(token):
    try:
        _active.reset(token)
    except ValueError:  # async generator finalized from a different context
        pass
