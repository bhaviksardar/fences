"""
agentfences.openai_agents against the real OpenAI Agents SDK, with a fake model that
returns tool calls and usage. Offline, no API keys. Python 3.10+.

    pip install ./sdk openai-agents && python tests/test_openai_agents.py
"""
import asyncio
import math

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

import agentfences
from agentfences import FencesStop, governed, get_active_run, openai_agents
from agentfences.pricing import price_of

import test_local  # its RecordingClient stands in for the server (and it blocks the network)
from test_langchain import BudgetServer, cloud_with

set_tracing_disabled(True)  # the Agents SDK would otherwise export traces to OpenAI

P = price_of("gpt-4o")
PER_CALL = 600 * P["in"] + 400 * P["cache_read"] + 200 * P["out"]


class FakeModel(Model):
    """Calls the lookup tool `tool_turns` times, then answers."""
    def __init__(self, tool_turns=1000):
        self.turns, self.tool_turns = 0, tool_turns
        self.model = "gpt-4o"  # like OpenAI's own Model classes, so the call can be priced

    async def get_response(self, *args, **kwargs):
        self.turns += 1
        if self.turns <= self.tool_turns:
            output = [ResponseFunctionToolCall(type="function_call", id=f"f{self.turns}", call_id=f"c{self.turns}",
                                               name="lookup", arguments=f'{{"query": "q{self.turns}"}}')]
        else:
            output = [ResponseOutputMessage(type="message", id="m", role="assistant", status="completed",
                                            content=[ResponseOutputText(type="output_text", text="done", annotations=[])])]
        usage = Usage(requests=1, input_tokens=1000, output_tokens=200, total_tokens=1200,
                      input_tokens_details=InputTokensDetails(cached_tokens=400, cache_write_tokens=0),
                      output_tokens_details=OutputTokensDetails(reasoning_tokens=0))
        return ModelResponse(output=output, usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@function_tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


def research_agent(tool_turns=1000):
    return Agent(name="research_agent", instructions="Research.", tools=[lookup], model=FakeModel(tool_turns))


def check_stops_on_budget():
    sent = cloud_with(BudgetServer(PER_CALL * 3.5))
    try:
        try:
            asyncio.run(openai_agents.run(research_agent(), "cats", budget_usd=PER_CALL * 3.5, max_turns=1000))
        except FencesStop as stop:
            assert stop.breach_type == "budget_exceeded" and "budget limit" in stop.message
        else:
            raise AssertionError("the agent should have been stopped")
        agentfences.flush()
        assert [p["agent_name"] for p in sent.of("start")] == ["research_agent"]
        assert sent.of("end")[-1]["status"] == "breached", sent.of("end")
        spans = [e["attributes"] for e in sent.of("event")]
        llm = [a for a in spans if a["gen_ai.operation.name"] == "chat"]
        tools = [a for a in spans if a["gen_ai.operation.name"] == "execute_tool"]
        assert len(llm) == 4 and all(math.isclose(a["fences.cost_usd"], PER_CALL) for a in llm), llm
        assert llm[0]["gen_ai.request.model"] == "gpt-4o" and llm[0]["gen_ai.provider.name"] == "openai"
        assert llm[0]["gen_ai.usage.input_tokens"] == 1000 and llm[0]["gen_ai.usage.cache_read.input_tokens"] == 400
        assert len(tools) == 3 and tools[0]["gen_ai.tool.name"] == "lookup" and tools[0]["gen_ai.tool.call.arguments"] == '{"query": "q1"}'
        assert tools[0]["gen_ai.tool.call.id"] == "c1", tools[0]
    finally:
        agentfences.init(local_only=True)


def check_success():
    sent = test_local.cloud()
    try:
        result = asyncio.run(openai_agents.run(research_agent(tool_turns=2), "cats", agent_name="cat_researcher", budget_usd=10))
        assert result.final_output == "done"
        agentfences.flush()
        assert [p["agent_name"] for p in sent.of("start")] == ["cat_researcher"] and sent.of("end")[-1]["status"] == "success"
        assert [e["attributes"]["gen_ai.operation.name"] for e in sent.of("event")] == ["chat", "execute_tool", "chat", "execute_tool", "chat"]
    finally:
        agentfences.init(local_only=True)


def check_hooks_inside_governed():
    @governed(budget_usd=10)
    async def agent():
        await Runner.run(research_agent(tool_turns=1), "cats", hooks=openai_agents.FencesRunHooks())
        run = get_active_run()
        return run.iterations, math.isclose(run.cost_usd, PER_CALL * 2)
    assert asyncio.run(agent()) == (2, True)

    @governed(max_iterations=2)
    async def stopped():
        await Runner.run(research_agent(), "cats", hooks=openai_agents.FencesRunHooks(), max_turns=100)
    try:
        asyncio.run(stopped())
    except FencesStop as stop:
        assert stop.breach_type == "iteration_limit"
    else:
        raise AssertionError("expected FencesStop")


def check_custom_hooks_rejected():
    try:
        asyncio.run(openai_agents.run(research_agent(), "cats", hooks=object()))
    except TypeError as e:
        assert "FencesRunHooks" in str(e)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    agentfences.init(local_only=True)
    for check in [check_stops_on_budget, check_success, check_hooks_inside_governed, check_custom_hooks_rejected]:
        check()
        print("ok", check.__name__)
    print("openai-agents checks passed")
