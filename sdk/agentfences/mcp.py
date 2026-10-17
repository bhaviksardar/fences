"""
MCP: record every tool call an MCP client session makes, and guard the ones you name.

    session = agentfences.mcp.wrap(session, guard=["refund", "send_email"], cost_usd={"refund": lambda a: a["amount"]})

Each `await session.call_tool(name, arguments)` inside a governed run becomes a tool span,
with no decorator per tool. A result with isError counts as a failed call. Guarded tools go
through the same check and settle as @tool(guard=True); a refused one raises ToolBlocked and
never reaches the MCP server. Outside a governed run calls pass straight through.
"""
from typing import Callable, Iterable, Mapping, Union

from .core import get_active_run
from .tools import call_async, items_summary


class MCPToolError(Exception):
    """An MCP tool answered with isError; recorded as the call's error, the result still goes back."""
    def __init__(self, result):
        super().__init__(" ".join(getattr(c, "text", "") for c in getattr(result, "content", None) or ()).strip() or "isError")
        self.result = result


def wrap(session, *, guard: Union[bool, Iterable[str]] = (), cost_usd: Mapping[str, Union[float, Callable[[dict], float]]] = None,
         approve: Iterable[str] = ()):
    """
    Patch session.call_tool in place and return the session (an mcp ClientSession, or anything with that method).

    guard: tool names to check with Fences before each call, or True for every tool.
    cost_usd: per tool, what a call costs the run, a number or a function of its arguments; such a tool is guarded.
    approve: tool names a person approves from the dashboard first; such a tool is guarded.
    """
    names = lambda v: {v} if isinstance(v, str) else set(v or ())  # guard="refund" is one name, not six letters
    original, costs, approve = session.call_tool, dict(cost_usd or {}), names(approve)
    named = set() if guard is True else names(guard)

    async def call_tool(name, arguments=None, *args, **kwargs):
        run = get_active_run()
        if run is None:
            return await original(name, arguments, *args, **kwargs)
        c = costs.get(name)
        guarded = guard is True or name in named or name in costs or name in approve
        cost = guarded and (lambda: (c(arguments or {}) if callable(c) else c) or 0)

        async def call():
            result = await original(name, arguments, *args, **kwargs)
            if getattr(result, "isError", False):
                raise MCPToolError(result)
            return result
        try:
            return await call_async(run, name, items_summary((arguments or {}).items()), cost, name in approve, call)
        except MCPToolError as e:
            return e.result

    session.call_tool = call_tool
    return session
