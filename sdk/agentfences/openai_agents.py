"""
Fences for the OpenAI Agents SDK (Python 3.10+), with no decorators:

    from agentfences import openai_agents, FencesStop

    try:
        result = await openai_agents.run(agent, "Research cats", budget_usd=2)
    except FencesStop as stop:
        print(stop.message)   # why it stopped, ready to show the user

run() is Runner.run governed by Fences: each model call is prechecked against the budget
before it's sent, then recorded with its tokens and cost and checkpointed, so limits
apply, and each tool call is recorded. Already inside
@governed? Pass FencesRunHooks() as Runner.run(..., hooks=...) instead. A sub-agent's run
(Agent.as_tool, or Runner.run in a tool) is governed too, with the same budget and limits,
unless it's given hooks of its own: those should subclass FencesRunHooks.
"""
import contextvars
import json
import time
from typing import Any, Optional

from agents import RunHooks, Runner
from agents.models import get_default_model

from . import core, precheck
from .exceptions import FencesStop
from .spans import names_of, record_llm_call, record_tool_call


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
        self._holds: dict = {}    # model call key -> its budget hold

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        settings = getattr(agent, "model_settings", None)
        params = {"max_tokens": getattr(settings, "max_tokens", None)}
        # RunHooks has no model-error hook: a hold still here belongs to a call that failed, so it was never billed
        precheck.finish(self._holds.pop(("llm", id(agent)), None), failed=True)
        # Before the request goes out: refuse it (FencesStop) if it might not fit the budget
        self._holds[("llm", id(agent))] = await precheck.hold_async(_model_name(agent), [system_prompt, input_items], params)
        self._started[("llm", id(agent))] = time.perf_counter()

    async def on_llm_end(self, context, agent, response) -> None:
        if core.get_active_run() is None:
            return
        t0 = self._started.pop(("llm", id(agent)), time.perf_counter())
        precheck.finish(self._holds.pop(("llm", id(agent)), None))  # the checkpoint below swaps it for the actual cost
        source = {"model": _model_name(agent), "usage": response.usage}  # Responses-API-shaped usage
        record_llm_call("openai", source["model"], t0, source, integration="openai-agents")
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
        try:
            names = names_of(json.loads(args)) if args else None  # the model's arguments, as JSON
        except ValueError:
            names = None
        record_tool_call(getattr(context, "tool_name", None) or getattr(tool, "name", "tool"), args, t0,
                         call_id=getattr(context, "tool_call_id", None), integration="openai-agents", arg_names=names)


async def run(agent, input: Any, *, agent_name: Optional[str] = None, budget_usd: Optional[float] = None,
              max_iterations: Optional[int] = None, max_duration_ms: Optional[int] = None,
              max_tokens: Optional[int] = None, on_breach: Optional[str] = None, **runner_kwargs):
    """
    Runner.run(agent, input, **runner_kwargs), governed as one Fences run named after the
    agent. Limits and on_breach work as in @governed and are all optional. Raises FencesStop when a
    limit is crossed. For hooks of your own, subclass FencesRunHooks (calling super())
    and pass it to Runner.run inside @governed.
    """
    if "hooks" in runner_kwargs:
        raise TypeError("openai_agents.run() supplies its own hooks; subclass FencesRunHooks "
                        "and use Runner.run(..., hooks=...) inside @governed instead")

    async def governed_run():
        return await Runner.run(agent, input, hooks=FencesRunHooks(), **runner_kwargs)
    governed_run.__name__ = agent_name or getattr(agent, "name", None) or "openai_agent"  # the run's agent name
    return await core.governed(budget_usd, max_iterations, max_duration_ms, max_tokens, on_breach)(governed_run)()


# A run governed by FencesRunHooks is in progress in this context: Runner.run with no hooks of its
# own inside it (Agent.as_tool's sub-agent, or a tool calling Runner.run) gets them, so the
# sub-agent's model calls are prechecked and charged to the same Fences run.
_governing = contextvars.ContextVar("agentfences_openai_agents_governing", default=False)


def _hooks_for(kwargs: dict) -> dict:
    if kwargs.get("hooks") is None and _governing.get() and core.get_active_run() is not None:
        kwargs["hooks"] = FencesRunHooks()  # its own: hold keys are per agent, and the parent's run concurrently
    return kwargs


def _patch_runner():
    if getattr(Runner.run, "_fences", False):
        return
    run_, run_streamed = Runner.run, Runner.run_streamed

    async def governed_run(cls, *args, **kwargs):
        kwargs = _hooks_for(kwargs)
        token = _governing.set(True) if isinstance(kwargs.get("hooks"), FencesRunHooks) else None
        try:
            return await run_(*args, **kwargs)
        finally:
            if token is not None:
                _governing.reset(token)

    def governed_streamed(cls, *args, **kwargs):
        return run_streamed(*args, **_hooks_for(kwargs))

    governed_run._fences = governed_streamed._fences = True
    Runner.run, Runner.run_streamed = classmethod(governed_run), classmethod(governed_streamed)


_patch_runner()
