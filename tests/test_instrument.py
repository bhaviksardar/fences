"""
agentfences.instrument() against the real openai and anthropic clients, with their HTTP
replaced by canned responses: no network, no API keys.

    pip install ./sdk openai anthropic && python tests/test_instrument.py
"""
import asyncio
import json
import math

import anthropic
import anthropic._base_client
import openai
import openai._base_client

import agentfences
from agentfences import governed, get_active_run
from agentfences.pricing import price_of

CHAT = {"id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4o-2024-08-06",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "prompt_tokens_details": {"cached_tokens": 400}}}
RESPONSE = {"id": "r", "object": "response", "created_at": 0, "model": "gpt-4o", "output": [], "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [], "status": "completed",
            "usage": {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
MESSAGE = {"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929", "content": [],
           "stop_reason": "end_turn", "stop_sequence": None,
           "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}}


def http(sdk_base_client):
    """The HTTP library this SDK version is built on: httpx2 in current releases, httpx before."""
    return getattr(sdk_base_client, "httpx2", None) or sdk_base_client.httpx


def handler(request):
    lib = http(openai._base_client)
    body = json.loads(request.content or b"{}")
    if body.get("model") == "rate-limited":
        return lib.Response(429, json={"error": {"message": "slow down", "type": "rate_limit_error"}})
    path = request.url.path
    data = CHAT if path.endswith("/chat/completions") else RESPONSE if path.endswith("/responses") else MESSAGE
    return lib.Response(200, json=data)


def clients():
    o, a = http(openai._base_client), http(anthropic._base_client)
    kw = dict(api_key="test", max_retries=0)
    return (openai.OpenAI(http_client=o.Client(transport=o.MockTransport(handler)), **kw),
            openai.AsyncOpenAI(http_client=o.AsyncClient(transport=o.MockTransport(handler)), **kw),
            anthropic.Anthropic(http_client=a.Client(transport=a.MockTransport(handler)), **kw),
            anthropic.AsyncAnthropic(http_client=a.AsyncClient(transport=a.MockTransport(handler)), **kw))


def cost(model, inp=0, cache_read=0, out=0):
    p = price_of(model)
    return inp * p["in"] + cache_read * p.get("cache_read", p["in"]) + out * p["out"]


def main():
    agentfences.init(local_only=True, instrument=True)
    assert len(agentfences.instrument()) == 6  # all six classes patched, and patching again is harmless
    oa, aoa, an, aan = clients()
    msgs = [{"role": "user", "content": "hi"}]

    oa.chat.completions.create(model="gpt-4o", messages=msgs)  # outside a run: nothing recorded, nothing breaks

    @governed(budget_usd=1)
    async def agent():
        oa.chat.completions.create(model="gpt-4o", messages=msgs)
        await aoa.responses.create(model="gpt-4o", input="hi")
        an.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=msgs)
        await aan.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=msgs)
        try:
            oa.chat.completions.create(model="rate-limited", messages=msgs)
        except openai.RateLimitError:
            pass  # the provider's error still reaches the agent
        run = get_active_run()
        return list(run.events), run.cost_usd, run.iterations

    spans, spent, iterations = asyncio.run(agent())
    a = [s["attributes"] for s in spans]
    assert [(x["gen_ai.provider.name"], s["status"]) for x, s in zip(a, spans)] == [
        ("openai", "ok"), ("openai", "ok"), ("anthropic", "ok"), ("anthropic", "ok"), ("openai", "error")], spans
    assert all(x["gen_ai.operation.name"] == "chat" and s["end_ts"] >= s["start_ts"] for x, s in zip(a, spans))

    chat, resp, msg, amsg, failed = a
    assert spans[0]["name"] == "chat gpt-4o-2024-08-06" and chat["gen_ai.request.model"] == "gpt-4o"
    assert (chat["gen_ai.usage.input_tokens"], chat["gen_ai.usage.cache_read.input_tokens"], chat["gen_ai.usage.output_tokens"]) == (1000, 400, 200)
    assert math.isclose(chat["fences.cost_usd"], cost("gpt-4o", inp=600, cache_read=400, out=200))
    assert math.isclose(resp["fences.cost_usd"], cost("gpt-4o", inp=500, out=100))
    assert msg == amsg  # sync and async record the same
    assert msg["gen_ai.usage.input_tokens"] == 1100  # Anthropic reports cached input separately; OTel totals it
    assert math.isclose(msg["fences.cost_usd"], cost("claude-sonnet-4-5", inp=100, cache_read=1000, out=50))
    assert failed["http.response.status_code"] == 429 and failed["error.type"] == "RateLimitError"
    assert "gen_ai.usage.input_tokens" not in failed and spans[4]["name"] == "chat rate-limited"
    assert (spent, iterations) == (0, 0), "recording doesn't checkpoint or add spend"
    print("instrument checks passed")


if __name__ == "__main__":
    main()
