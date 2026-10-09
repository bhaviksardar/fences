"""
Fences for LangChain and LangGraph, with no decorators:

    from agentfences.langchain import FencesCallbackHandler
    from agentfences import FencesStop

    try:
        graph.invoke(inputs, config={"callbacks": [FencesCallbackHandler(budget_usd=2)]})
    except FencesStop as stop:
        print(stop.message)   # why it stopped, ready to show the user

The outermost chain or graph run becomes a Fences run (or joins one already started by
@governed). Each model call is recorded with its tokens and cost and checkpointed, so
limits apply; each tool call is recorded. Requires langchain-core.
"""
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from . import core
from .exceptions import FencesStop
from .pricing import read_usage
from .spans import record_llm_call, record_tool_call

log = logging.getLogger("agentfences")


def _response_of(result) -> Any:
    """The object carrying usage in an LLMResult: the chat message, or llm_output for plain LLMs."""
    for generations in result.generations or ():
        for g in generations:
            message = getattr(g, "message", None)
            if message is not None and read_usage(message):
                return message
    out = result.llm_output or {}
    usage = out.get("token_usage") or out.get("usage")
    return {"model": out.get("model_name") or out.get("model") or "", "usage": usage} if usage else None


class FencesCallbackHandler(BaseCallbackHandler):
    """
    Governs a LangChain or LangGraph run. Limits work as in @governed and are all
    optional; in cloud mode, limits left out come from the agent's page in the dashboard.
    agent_name defaults to the outermost run's name (e.g. "LangGraph").
    """
    raise_error = True  # so a FencesStop can halt the graph; anything else this handler hits is logged, never raised

    def __init__(self, agent_name: Optional[str] = None, budget_usd: Optional[float] = None,
                 max_iterations: Optional[int] = None, max_duration_ms: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        self.agent_name = agent_name
        self.limits = (budget_usd, max_iterations, max_duration_ms, max_tokens)
        self._runs: Dict[UUID, Any] = {}       # root LangChain run id -> (RunState, started_here)
        self._root: Dict[UUID, UUID] = {}      # any LangChain run id -> its root
        self._started: Dict[UUID, tuple] = {}  # model or tool call in progress -> (kind, name, args, t0)

    # ── run boundaries ────────────────────────────────────────────────────────

    def _enter(self, run_id: UUID, parent_run_id: Optional[UUID], name: str):
        if parent_run_id is not None:
            self._root[run_id] = self._root.get(parent_run_id, parent_run_id)
            return
        self._root[run_id] = run_id
        active = core.get_active_run()
        if active is not None:  # inside @governed: record into that run, don't start another
            self._runs[run_id] = (active, False)
            return
        core._require_init()
        run = core._start_run(self.agent_name or name or "langchain_agent", *self.limits)
        self._runs[run_id] = (run, True)

    def _exit(self, run_id: UUID, error: Optional[BaseException]):
        entry = self._runs.pop(run_id, None) if self._root.get(run_id) == run_id else None
        for child in [k for k, root in self._root.items() if root == run_id]:
            self._root.pop(child, None)
        if entry is None or not entry[1]:
            return
        run = entry[0]
        if isinstance(error, FencesStop):
            status, text = "breached", None
        else:
            status, text = core._outcome(run, error)
        core._end_run(run, status, text)

    def _run(self, run_id: UUID):
        entry = self._runs.get(self._root.get(run_id))
        return entry[0] if entry else None

    def _with_run(self, run_id: UUID, fn):
        """Call fn with this LangChain run's Fences run active, whichever thread LangChain uses."""
        run = self._run(run_id)
        if run is None:
            return None
        token = core._set_active_run(run)
        try:
            return fn(run)
        finally:
            core._reset_active_run(token)

    def _safely(self, fn):
        """Only FencesStop may escape: a bug here must never break the user's graph."""
        try:
            return fn()
        except FencesStop:
            raise
        except Exception:
            log.exception("agentfences: LangChain callback failed; the run continues unrecorded")

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or ""
        self._safely(lambda: self._enter(run_id, parent_run_id, name))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._safely(lambda: self._exit(run_id, None))

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._safely(lambda: self._exit(run_id, error))

    # ── model calls ───────────────────────────────────────────────────────────

    def _model_start(self, serialized, run_id, parent_run_id, kwargs):
        def go():
            if parent_run_id is None:  # a model invoked on its own is a run of its own
                self._enter(run_id, None, kwargs.get("name") or (serialized or {}).get("name") or "")
            else:
                self._enter(run_id, parent_run_id, "")
            params = kwargs.get("invocation_params") or {}
            provider = (kwargs.get("metadata") or {}).get("ls_provider") or "langchain"  # e.g. "openai", "anthropic"
            self._started[run_id] = ("llm", params.get("model") or params.get("model_name"), provider, time.perf_counter())
        self._safely(go)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._model_start(serialized, run_id, parent_run_id, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._model_start(serialized, run_id, parent_run_id, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        def record(run):
            _, model, provider, t0 = self._started.pop(run_id, ("llm", None, "langchain", time.perf_counter()))
            source = _response_of(response)
            record_llm_call(provider, model, t0, source, integration="langchain")
            # The model call is the step: count it, price it, check limits
            result = core.checkpoint_sync(source) if source is not None else core.checkpoint_sync()
            if result.breached:
                raise FencesStop(result)
        try:
            self._safely(lambda: self._with_run(run_id, record))
        finally:
            if self._root.get(run_id) == run_id:  # a model run on its own ends here
                self._safely(lambda: self._exit(run_id, None))

    def on_llm_error(self, error, *, run_id, **kwargs):
        def record(run):
            _, model, provider, t0 = self._started.pop(run_id, ("llm", None, "langchain", time.perf_counter()))
            record_llm_call(provider, model, t0, error=error, integration="langchain")
        self._safely(lambda: self._with_run(run_id, record))
        if self._root.get(run_id) == run_id:
            self._safely(lambda: self._exit(run_id, error))

    # ── tool calls ────────────────────────────────────────────────────────────

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        def go():
            self._enter(run_id, parent_run_id, "")
            args = ", ".join(f"{k}={v!r}" for k, v in inputs.items()) if isinstance(inputs, dict) else str(input_str)
            name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
            self._started[run_id] = ("tool", name, args, time.perf_counter())
        self._safely(go)

    def _tool_end(self, run_id, error):
        def record(run):
            started = self._started.pop(run_id, None)
            if started is None:
                return
            _, name, args, t0 = started
            record_tool_call(name, args, t0, error, integration="langchain")
        self._safely(lambda: self._with_run(run_id, record))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._tool_end(run_id, None)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._tool_end(run_id, error)
