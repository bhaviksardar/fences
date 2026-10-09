"""
Offline checks for agentfences in local mode: no server, no API key, no network.

    pip install ./sdk && python tests/test_local.py
"""
import asyncio
import logging
import math
import socket
from types import SimpleNamespace as NS

import agentfences
from agentfences import (
    governed, checkpoint, checkpoint_sync, log_decision, get_active_run,
    BudgetExceeded, IterationLimitReached, TimeLimitReached, TokenLimitReached,
)
from agentfences.pricing import price_of


def no_network(*args, **kwargs):
    raise AssertionError("local mode attempted a network call")


socket.socket.connect = no_network
socket.create_connection = no_network


def check_requires_init():
    @governed(budget_usd=1)
    def agent():
        return "ran"
    try:
        agent()
    except RuntimeError as e:
        assert "init()" in str(e)
    else:
        raise AssertionError("a governed call before init() should raise")


def check_readme_quickstart():
    @governed(budget_usd=0.10, max_iterations=5, max_tokens=2000)
    async def my_agent(query: str):
        for step in range(100):
            log_decision(reasoning=f"step {step}: searching", action="search")
            result = await checkpoint(cost_delta_usd=0.02, tokens_used=200)
            if result.breached:
                return result.message
        return "done"

    assert asyncio.run(my_agent("hello")) == (
        "I've reached my budget limit (spent $0.1200 of $0.1000). I'll summarize what I found so far."
    )


def steps_until_breach(**limits):
    """Run checkpoints until one breaches; return (completed steps, the breaching result)."""
    @governed(**limits)
    async def agent():
        for step in range(1000):
            result = await checkpoint(cost_delta_usd=0.02, tokens_used=100)
            if result.breached:
                return step, result
        raise AssertionError("no limit tripped")
    return asyncio.run(agent())


def check_limits_trip_once_exceeded():
    steps, r = steps_until_breach(budget_usd=0.10)  # exactly $0.10 is allowed
    assert (steps, r.breach_type) == (5, "budget_exceeded"), (steps, r.breach_type)

    steps, r = steps_until_breach(budget_usd=99, max_iterations=5)
    assert (steps, r.breach_type) == (5, "iteration_limit"), (steps, r.breach_type)

    steps, r = steps_until_breach(budget_usd=99, max_tokens=300)
    assert (steps, r.breach_type) == (3, "token_limit"), (steps, r.breach_type)

    assert r.breached and not r.ok and r.message and r.system_prompt


def check_time_limit():
    @governed(budget_usd=99, max_duration_ms=100)
    async def slow():
        for _ in range(100):
            await asyncio.sleep(0.03)
            result = await checkpoint()
            if result.breached:
                return result.breach_type
    assert asyncio.run(slow()) == "time_limit"


def check_raise_on_breach():
    cases = [
        (dict(budget_usd=0.05), dict(cost_delta_usd=0.02), BudgetExceeded),
        (dict(budget_usd=99, max_iterations=2), {}, IterationLimitReached),
        (dict(budget_usd=99, max_tokens=10), dict(tokens_used=6), TokenLimitReached),
    ]
    for limits, step, expected in cases:
        @governed(raise_on_breach=True, **limits)
        async def agent():
            for _ in range(100):
                await checkpoint(**step)
        try:
            asyncio.run(agent())
        except expected:
            pass
        else:
            raise AssertionError(f"expected {expected.__name__}")

    @governed(budget_usd=99, max_duration_ms=50, raise_on_breach=True)
    async def slow():
        while True:
            await asyncio.sleep(0.03)
            await checkpoint()
    try:
        asyncio.run(slow())
    except TimeLimitReached as e:
        assert e.duration_ms > e.max_duration_ms == 50
    else:
        raise AssertionError("expected TimeLimitReached")


def check_sync_agent():
    @governed(budget_usd=99, max_iterations=3)
    def agent():
        steps = 0
        while not checkpoint_sync(cost_delta_usd=0.01).breached:
            steps += 1
        return steps
    assert agent() == 3


def check_streaming_agent():
    @governed(budget_usd=99, max_iterations=2)
    async def stream():
        for i in range(10):
            if (await checkpoint()).breached:
                return
            yield i

    async def collect():
        return [x async for x in stream()]
    assert asyncio.run(collect()) == [0, 1]


def check_concurrent_runs_are_independent():
    @governed(budget_usd=0.10)
    async def agent():
        for step in range(100):
            await asyncio.sleep(0)  # interleave with the other runs
            if (await checkpoint(cost_delta_usd=0.02)).breached:
                return step

    async def main():
        return await asyncio.gather(*(agent() for _ in range(5)))
    assert asyncio.run(main()) == [5] * 5  # a shared budget would stop them much sooner


def check_nested_runs():
    @governed(budget_usd=99)
    async def inner():
        await checkpoint()
        return get_active_run().run_id

    @governed(budget_usd=99)
    async def outer():
        outer_id = get_active_run().run_id
        inner_id = await inner()
        assert get_active_run().run_id == outer_id != inner_id
        await checkpoint()
        return get_active_run().iterations
    assert asyncio.run(outer()) == 1  # the inner run's checkpoint didn't count here


def check_outside_a_run():
    assert get_active_run() is None
    assert asyncio.run(checkpoint(cost_delta_usd=5)).ok
    log_decision("ignored outside a run")


def priced(model, inp=0, cache_read=0, cache_write=0, out=0):
    """Expected cost from the bundled table, so these checks survive price refreshes."""
    p = price_of(model)
    assert p, f"{model} missing from prices.json"
    if "above" in p and inp + cache_read + cache_write > p["above"]:
        p = dict(p, **p["tier"])
    return inp * p["in"] + cache_read * p.get("cache_read", p["in"]) + cache_write * p.get("cache_write", p["in"]) + out * p["out"]


def step(response, **extra):
    """Run one checkpoint on a response; return (spent_usd, tokens)."""
    @governed(budget_usd=1e9)
    def agent():
        result = checkpoint_sync(response, **extra)
        return result.spent_usd, get_active_run().tokens_used
    return agent()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def check_prices_each_response_shape():
    openai_chat = NS(model="gpt-4o-2024-08-06", usage=NS(
        prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=NS(cached_tokens=400)))
    spent, tokens = step(openai_chat)
    assert close(spent, priced("gpt-4o", inp=600, cache_read=400, out=200)) and tokens == 1200, (spent, tokens)

    openai_responses = {"model": "gpt-4o", "usage": {"input_tokens": 1000, "output_tokens": 200,
                        "input_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 500}}}
    spent, tokens = step(openai_responses)
    assert close(spent, priced("gpt-4o", inp=200, cache_read=300, cache_write=500, out=200)) and tokens == 1200

    anthropic = NS(model="claude-sonnet-4-5-20250929", usage=NS(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=1000, cache_creation_input_tokens=500))
    spent, tokens = step(anthropic)
    assert close(spent, priced("claude-sonnet-4-5", inp=100, cache_read=1000, cache_write=500, out=50)) and tokens == 1650

    long_context = NS(model="claude-sonnet-4-5", usage=NS(
        input_tokens=250_000, output_tokens=1000, cache_read_input_tokens=None, cache_creation_input_tokens=None))
    spent, _ = step(long_context)
    assert close(spent, priced("claude-sonnet-4-5", inp=250_000, out=1000))
    assert spent > priced("claude-sonnet-4-5", inp=200_000, out=1000) * 1.25  # the long-context tier applied

    gemini = NS(model_version="gemini-2.5-flash", usage_metadata=NS(
        prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=300, cached_content_token_count=None))
    spent, tokens = step(gemini)
    assert close(spent, priced("gemini-2.5-flash", inp=1000, out=400)) and tokens == 1400  # thinking billed as output

    langchain = NS(response_metadata={"model_name": "gpt-4o"}, usage_metadata={
        "input_tokens": 1000, "output_tokens": 100, "input_token_details": {"cache_read": 200}})
    spent, tokens = step(langchain)
    assert close(spent, priced("gpt-4o", inp=800, cache_read=200, out=100)) and tokens == 1100

    gateway = {"model": "anything", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0123}}
    assert step(gateway) == (0.0123, 15)  # a reported cost wins over the table


def check_extra_costs_and_old_calls():
    resp = NS(model="gpt-4o", usage=NS(prompt_tokens=1000, completion_tokens=0, prompt_tokens_details=None))
    spent, tokens = step(resp, cost_delta_usd=0.01, tokens_used=5)  # e.g. a paid tool call on top
    assert close(spent, priced("gpt-4o", inp=1000) + 0.01) and tokens == 1005

    @governed(budget_usd=1)
    def old_style():
        checkpoint_sync(0.02, 450)  # pre-0.1.7 positional form still works
        run = get_active_run()
        return run.cost_usd, run.tokens_used
    assert old_style() == (0.02, 450)


def check_budget_stops_on_real_usage():
    resp = NS(model="gpt-4o", usage=NS(prompt_tokens=10_000, completion_tokens=1000, prompt_tokens_details=None))
    per_call = priced("gpt-4o", inp=10_000, out=1000)

    @governed(budget_usd=per_call * 3.5)
    async def agent():
        for calls in range(100):
            result = await checkpoint(resp)
            if result.breached:
                return calls, result
    calls, result = asyncio.run(agent())
    assert calls == 3 and result.breach_type == "budget_exceeded" and close(result.spent_usd, per_call * 4)


def check_unknown_model_warns_and_custom_prices():
    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    try:
        resp = {"model": "my-finetune-v2", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}
        assert step(resp) == (0.0, 2_000_000)  # tokens still counted
        step(resp)
        assert len([m for m in seen if "no price for model 'my-finetune-v2'" in m]) == 1  # warned once

        assert step(NS(model="gpt-4o")) == (0.0, 0)
        assert any("has no token usage" in m for m in seen)
    finally:
        logging.getLogger("agentfences").removeHandler(handler)

    agentfences.init(local_only=True, prices={"my-finetune-v2": {"input": 2.0, "output": 6.0}})
    spent, _ = step(resp)
    assert close(spent, 8.0)  # 1M in at $2 + 1M out at $6


def check_breach_messages():
    # Server-side stop reasons get their own wording, so an agent can tell the user why it stopped
    from agentfences.core import _make_breach_result
    for breach in ("stopped_by_user", "key_daily_budget", "key_monthly_budget", "fences_unreachable"):
        result = _make_breach_result(breach)
        assert result.breached and result.breach_type == breach
        assert "Governance limit reached" not in result.message, breach
        assert "governance limit has been reached" not in result.system_prompt, breach


class RecordingClient:
    """Stands in for the backend client and keeps every payload the SDK would send."""
    def __init__(self):
        self.sent = []

    def start_run(self, run_id, agent_name, *limits, context=None):
        self.sent.append(("start", {"agent_name": agent_name, "context": context}))
        return {"ok": True}

    def checkpoint(self, *args):
        return {"ok": True}

    def log_decision(self, run_id, iteration, reasoning, action):
        self.sent.append(("decision", {"reasoning": reasoning, "action": action}))

    def end_run(self, run_id, status, error=None, exception=None):
        self.sent.append(("end", {"status": status, "error": error, "exception": exception}))

    def of(self, kind):
        return [payload for k, payload in self.sent if k == kind]


def cloud(**init_args):
    """Cloud-mode init with the network replaced by a RecordingClient."""
    agentfences.init(api_key="fc_test", endpoint="http://fences.invalid", **init_args)
    agentfences.core._client = recorder = RecordingClient()
    return recorder


def check_run_context():
    sent = cloud(environment="prod", release="v1.4.2")
    try:
        @governed(budget_usd=1)
        async def agent():
            return dict(get_active_run().context)

        with agentfences.context(user_id="u_1", session_id="s_1"):
            with agentfences.context(session_id="s_2", trace_id=None):  # inner wins; None is left out
                inner = asyncio.run(agent())
            outer = asyncio.run(agent())
        bare = asyncio.run(agent())
        assert inner == {"environment": "prod", "release": "v1.4.2", "user_id": "u_1", "session_id": "s_2"}, inner
        assert outer == {"environment": "prod", "release": "v1.4.2", "user_id": "u_1", "session_id": "s_1"}, outer
        assert bare == {"environment": "prod", "release": "v1.4.2"}
        assert [p["context"] for p in sent.of("start")] == [inner, outer, bare]  # and that's what the backend got
    finally:
        agentfences.init(local_only=True)


def check_exception_capture():
    sent = cloud()
    try:
        def fetch_page():
            raise ValueError("page 7 returned 403")

        @governed(budget_usd=1)
        def agent():
            fetch_page()
        try:
            agent()
        except ValueError:
            pass
        else:
            raise AssertionError("the agent's exception must propagate")
        end = sent.of("end")[-1]
        assert end["status"] == "error" and end["error"] == "ValueError: page 7 returned 403", end
        e = end["exception"]
        assert e["type"] == "ValueError" and e["message"] == "page 7 returned 403"
        assert e["stack"][-1].endswith("in fetch_page") and any(f.endswith("in agent") for f in e["stack"]), e["stack"]

        @governed(budget_usd=1)
        def ok_agent():
            return "fine"
        ok_agent()
        assert sent.of("end")[-1] == {"status": "success", "error": None, "exception": None}
    finally:
        agentfences.init(local_only=True)


def check_redaction():
    def scrub(event):
        if event.get("action") == "internal":
            return None  # drop this decision entirely
        if event["type"] == "decision":
            event["reasoning"] = event["reasoning"].replace("alice@example.com", "[email]")
        if event["type"] == "run_start":
            event["context"] = {k: v for k, v in event["context"].items() if k != "user_id"}
        if event["type"] == "run_end" and event["exception"]:
            event["error"] = event["exception"]["type"]
            event["exception"] = dict(event["exception"], message="[redacted]")
        return event

    sent = cloud(redact=scrub, environment="prod")
    try:
        @governed(budget_usd=1)
        def agent():
            log_decision("emailing alice@example.com the report", action="send_email")
            log_decision("checking the internal admin panel", action="internal")
            raise RuntimeError("SMTP rejected alice@example.com")
        with agentfences.context(user_id="alice"):
            try:
                agent()
            except RuntimeError:
                pass
        agentfences.flush()
        assert sent.of("start")[-1]["context"] == {"environment": "prod"}
        assert sent.of("decision") == [{"reasoning": "emailing [email] the report", "action": "send_email"}]
        end = sent.of("end")[-1]
        assert end["error"] == "RuntimeError" and end["exception"]["message"] == "[redacted]"
        assert "alice" not in repr(sent.sent)
    finally:
        agentfences.init(local_only=True)

    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    for broken in (lambda event: 1 / 0, lambda event: "not a dict"):
        sent = cloud(redact=broken)
        try:
            @governed(budget_usd=1)
            def leaky():
                log_decision("secret plans")  # must not crash the agent, and must not be sent
                return "done"
            assert leaky() == "done"
            agentfences.flush()
            assert sent.of("decision") == [], sent.sent
            assert sent.of("start") and sent.of("end"), "runs still start and end"
        finally:
            agentfences.init(local_only=True)
    logging.getLogger("agentfences").removeHandler(handler)
    assert len([m for m in seen if "redact hook failed" in m]) == 2, seen  # once per init


if __name__ == "__main__":
    check_requires_init()
    check_breach_messages()
    agentfences.init(local_only=True)
    checks = [
        check_readme_quickstart, check_limits_trip_once_exceeded, check_time_limit,
        check_raise_on_breach, check_sync_agent, check_streaming_agent,
        check_concurrent_runs_are_independent, check_nested_runs, check_outside_a_run,
        check_prices_each_response_shape, check_extra_costs_and_old_calls, check_budget_stops_on_real_usage,
        check_unknown_model_warns_and_custom_prices, check_run_context, check_exception_capture, check_redaction,
    ]
    for check in checks:
        check()
        print("ok", check.__name__)
    agentfences.flush()
    print("all local checks passed")
