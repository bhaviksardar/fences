"""
FencesCallbackHandler against a real LangGraph graph: a fake chat model that returns
messages with usage, and a real LangChain tool. Offline, no API keys.

    pip install ./sdk langchain-core langgraph && python tests/test_langchain.py
"""
import asyncio
import math
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

import agentfences
from agentfences import FencesStop, governed, get_active_run
from agentfences.langchain import FencesCallbackHandler
from agentfences.pricing import price_of

import test_local  # its RecordingClient stands in for the server (and it blocks the network)

USAGE = {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "input_token_details": {"cache_read": 400}}
P = price_of("gpt-4o")
PER_CALL = 600 * P["in"] + 400 * P["cache_read"] + 200 * P["out"]


def reply(text="thinking"):
    return AIMessage(content=text, usage_metadata=USAGE, response_metadata={"model_name": "gpt-4o"})


@tool
def lookup(query: str) -> str:
    """Look something up."""
    if query == "blocked":
        raise PermissionError("403 from search API")
    return f"result for {query}"


class State(TypedDict):
    messages: Annotated[list, operator.add]
    steps: int


def build_graph(max_steps=1000, failing_tool=False):
    model = FakeMessagesListChatModel(responses=[reply()])

    def agent(state, config):
        return {"messages": [model.invoke(state["messages"], config)], "steps": state["steps"] + 1}

    def tools(state, config):
        try:
            out = lookup.invoke({"query": "blocked" if failing_tool else f"q{state['steps']}"}, config)
        except PermissionError as e:
            out = str(e)
        return {"messages": [HumanMessage(content=out)]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", lambda s: END if s["steps"] >= max_steps else "tools")
    g.add_edge("tools", "agent")
    return g.compile()


class BudgetServer(test_local.RecordingClient):
    """A recording server that also enforces a run budget, like the real one."""
    def __init__(self, budget):
        super().__init__()
        self.budget, self.spent = budget, 0.0

    def checkpoint(self, run_id, cost_delta_usd, *args):
        self.spent += cost_delta_usd
        if round(self.spent, 9) > self.budget:
            return {"ok": False, "breach": "budget_exceeded", "spent_usd": self.spent, "budget_usd": self.budget}
        return {"ok": True, "spent_usd": self.spent}


def cloud_with(server):
    agentfences.init(api_key="fc_test", endpoint="http://fences.invalid")
    agentfences.core._client = agentfences.control._client = server
    return server


START = {"messages": [HumanMessage(content="research cats")], "steps": 0}
CONFIG = {"recursion_limit": 10_000}


def check_graph_stops_on_budget():
    sent = cloud_with(BudgetServer(PER_CALL * 3.5))
    try:
        handler = FencesCallbackHandler(agent_name="research_graph", budget_usd=PER_CALL * 3.5)
        try:
            build_graph().invoke(START, {**CONFIG, "callbacks": [handler]})
        except FencesStop as stop:
            assert stop.breach_type == "budget_exceeded" and "budget limit" in stop.message and stop.system_prompt
        else:
            raise AssertionError("the graph should have been stopped")
        agentfences.flush()
        assert [p["agent_name"] for p in sent.of("start")] == ["research_graph"]
        assert sent.of("end")[-1]["status"] == "breached", sent.of("end")
        events = sent.of("event")
        llm = [e for e in events if e["type"] == "llm_call"]
        tools = [e for e in events if e["type"] == "tool_call"]
        assert len(llm) == 4 and all(math.isclose(e["cost_usd"], PER_CALL) for e in llm), llm  # the 4th call trips 3.5x
        assert llm[0]["model"] == "gpt-4o" and llm[0]["cache_read_tokens"] == 400
        assert len(tools) == 3 and tools[0]["name"] == "lookup" and tools[0]["args"] == "query='q1'" and tools[0]["ok"]
    finally:
        agentfences.init(local_only=True)


def check_success_and_tool_errors():
    sent = test_local.cloud()
    try:
        out = build_graph(max_steps=3, failing_tool=True).invoke(START, {**CONFIG, "callbacks": [FencesCallbackHandler(budget_usd=10)]})
        assert out["steps"] == 3
        agentfences.flush()
        assert sent.of("end")[-1]["status"] == "success"
        failed = [e for e in sent.of("event") if e["type"] == "tool_call"]
        assert failed and not failed[0]["ok"] and failed[0]["error"] == "PermissionError: 403 from search API", failed
    finally:
        agentfences.init(local_only=True)


def check_joins_a_governed_run():
    @governed(budget_usd=10)
    def agent():
        build_graph(max_steps=2).invoke(START, {**CONFIG, "callbacks": [FencesCallbackHandler()]})
        run = get_active_run()
        return run.iterations, math.isclose(run.cost_usd, PER_CALL * 2), [e["type"] for e in run.events]
    iterations, spent_right, kinds = agent()
    assert iterations == 2 and spent_right and kinds == ["llm_call", "tool_call", "llm_call"], (iterations, kinds)


def check_async_graph():
    async def main():
        try:
            await build_graph().ainvoke(START, {**CONFIG, "callbacks": [FencesCallbackHandler(max_iterations=3)]})
        except FencesStop as stop:
            return stop.breach_type
    assert asyncio.run(main()) == "iteration_limit"


def check_model_on_its_own():
    sent = test_local.cloud()
    try:
        model = FakeMessagesListChatModel(responses=[reply()])
        model.invoke("hi", config={"callbacks": [FencesCallbackHandler(agent_name="one_shot", budget_usd=1)]})
        agentfences.flush()
        assert [p["agent_name"] for p in sent.of("start")] == ["one_shot"] and sent.of("end")[-1]["status"] == "success"
    finally:
        agentfences.init(local_only=True)


if __name__ == "__main__":
    agentfences.init(local_only=True)
    for check in [check_graph_stops_on_budget, check_success_and_tool_errors, check_joins_a_governed_run,
                  check_async_graph, check_model_on_its_own]:
        check()
        print("ok", check.__name__)
    print("langchain checks passed")
