__version__ = "0.8.0"  # the one place the version lives; pyproject.toml reads it from here

from .core import init, governed, checkpoint, checkpoint_sync, log_decision, flush, get_active_run, CheckpointResult
from .events import context
from .tools import tool
from . import mcp
from .instrument import instrument
from .approvals import request_approval, request_approval_sync, Approval
from .exceptions import FencesError, FencesAuthError, AgentQuarantined, FencesStop, ToolBlocked

__all__ = [
    "init", "governed", "mcp", "checkpoint", "checkpoint_sync", "log_decision", "flush", "get_active_run", "context", "tool", "instrument",
    "request_approval", "request_approval_sync", "Approval",
    "CheckpointResult",
    "FencesError", "FencesAuthError", "AgentQuarantined", "FencesStop", "ToolBlocked",
]