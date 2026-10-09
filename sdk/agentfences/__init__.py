__version__ = "0.3.0"  # the one place the version lives; pyproject.toml reads it from here

from .core import init, governed, checkpoint, checkpoint_sync, log_decision, flush, get_active_run, CheckpointResult
from .events import context
from .tools import tool
from .instrument import instrument
from .control import request_approval, request_approval_sync, Approval
from .exceptions import FencesError, AgentQuarantined, FencesStop

__all__ = [
    "init", "governed", "checkpoint", "checkpoint_sync", "log_decision", "flush", "get_active_run", "context", "tool", "instrument",
    "request_approval", "request_approval_sync", "Approval",
    "CheckpointResult",
    "FencesError", "AgentQuarantined", "FencesStop",
]