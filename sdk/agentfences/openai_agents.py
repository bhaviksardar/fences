"""
Fences for the OpenAI Agents SDK (Python 3.10+), with no decorators:

    from agentfences import openai_agents, FencesStop

    try:
        result = await openai_agents.run(agent, "Research cats", budget_usd=2)
    except FencesStop as stop:
        print(stop.message)   # why it stopped, ready to show the user

run() is Runner.run governed by Fences: each model call is recorded with its tokens and
cost and checkpointed, so limits apply, and each tool call is recorded. Already inside
@governed? Pass FencesRunHooks() as Runner.run(..., hooks=...) instead.
"""
import time
from typing import Any, Optional

from agents import RunHooks, Runner
from agents.models import get_default_model

from . import core
from .exceptions import FencesStop
from .pricing import cost_of, read_usage
from .tools import MAX_ARGS


def _model_name(agent) -> str:
    model = getattr(agent, "model", None)
    if model is None:
        return get_default_model()  # what the Agents SDK runs when the agent names no model
    if isinstance(model, str):
        return model
    # A Model object: OpenAI's carry their model name; a custom one we can't price gets the usual warning
    return getattr(model, "model", None) or type(model).__name__


class FencesRunHooks(RunHooks):
    """Records model and tool calls into the active Fences run and checkpoints each model call."""

    def __init__(self):
        self._started: dict = {}  # tool call id or model call key -> start time

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._started[("llm", id(agent))] = time.perf_counter()

    async def on_llm_end(self, context, agent, response) -> None:
        if core.get_active_run() is None:
            return
        t0 = self._started.pop(("llm", id(agent)), time.perf_counter())
        source = {"model": _model_name(agent), "usage": response.usage}  # Responses-API-shaped usage
        usage = read_usage(source)
        event = {"type": "llm_call", "provider": "openai-agents", "model": source["model"], "ok": True,
                 "latency_ms": int((time.perf_counter() - t0) * 1000)}
        if usage:
            event.update(input_tokens=usage["input"], cache_read_tokens=usage["cache_read"],
                         cache_write_tokens=usage["cache_write"], output_tokens=usage["output"],
                         cost_usd=cost_of(source)[0])
        core.record_event(event)
        result = await core.checkpoint(source)  # the model call is the step: count it, price it, check limits
        if result.breached:
            raise FencesStop(result)

    async def on_tool_start(self, context, agent, tool) -> None:
        self._started[getattr(context, "tool_call_id", None) or ("tool", id(tool))] = time.perf_counter()

    async def on_tool_end(self, context, agent, tool, result) -> None:
        if core.get_active_run() is None:
            return
        t0 = self._started.pop(getattr(context, "tool_call_id", None) or ("tool", id(tool)), time.perf_counter())
        args = str(getattr(context, "tool_arguments", "") or "")
        core.record_event({"type": "tool_call", "name": getattr(context, "tool_name", None) or getattr(tool, "name", "tool"),
                           "args": args if len(args) <= MAX_ARGS else args[:MAX_ARGS - 1] + "…", "ok": True,
                           "error": None, "latency_ms": int((time.perf_counter() - t0) * 1000)})


async def run(agent, input: Any, *, agent_name: Optional[str] = None, budget_usd: Optional[float] = None,
              max_iterations: Optional[int] = None, max_duration_ms: Optional[int] = None,
              max_tokens: Optional[int] = None, **runner_kwargs):
    """
    Runner.run(agent, input, **runner_kwargs), governed as one Fences run named after the
    agent. Limits work as in @governed and are all optional. Raises FencesStop when a
    limit is crossed. For hooks of your own, subclass FencesRunHooks (calling super())
    and pass it to Runner.run inside @governed.
    """
    if "hooks" in runner_kwargs:
        raise TypeError("openai_agents.run() supplies its own hooks; subclass FencesRunHooks "
                        "and use Runner.run(..., hooks=...) inside @governed instead")

    async def governed_run():
        return await Runner.run(agent, input, hooks=FencesRunHooks(), **runner_kwargs)
    governed_run.__name__ = agent_name or getattr(agent, "name", None) or "openai_agent"  # the run's agent name
    return await core.governed(budget_usd, max_iterations, max_duration_ms, max_tokens)(governed_run)()
