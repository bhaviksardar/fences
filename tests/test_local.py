"""
Offline checks for agentfences in local mode: no server, no API key, no network.

    pip install ./sdk && python tests/test_local.py
"""
import asyncio
import os
import logging
import math
import socket
from types import SimpleNamespace as NS

import agentfences
from agentfences import (
    governed, checkpoint, checkpoint_sync, log_decision, get_active_run,
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


def check_removed_apis_fail_loudly():
    try:
        governed(budget_usd=1, raise_on_breach=True)  # removed in 0.3: checkpoint() never raises
    except TypeError:
        pass
    else:
        raise AssertionError("raise_on_breach should no longer be accepted")

    @governed(budget_usd=1)
    def agent():
        checkpoint_sync(0.02, 450)  # removed in 0.3: positional cost and tokens
    try:
        agent()
    except TypeError:
        pass  # a number isn't a model response, so it can't be priced
    else:
        raise AssertionError("positional cost should no longer be accepted")


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

    from agentfences.pricing import _prices
    tiered = next(m for m, p in sorted(_prices().items()) if "above" in p)  # any model with long-context pricing
    over = price_of(tiered)["above"] + 50_000
    long_context = NS(model=tiered, usage=NS(
        input_tokens=over, output_tokens=1000, cache_read_input_tokens=None, cache_creation_input_tokens=None))
    spent, _ = step(long_context)
    assert close(spent, priced(tiered, inp=over, out=1000))
    p = price_of(tiered)
    assert close(spent, over * p["tier"]["in"] + 1000 * p["tier"]["out"]), tiered  # the long-context tier applied

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
    from agentfences.results import breach_result
    for breach in ("stopped_by_user", "key_daily_budget", "key_monthly_budget", "fences_unreachable"):
        result = breach_result(breach)
        assert result.breached and result.breach_type == breach
        assert "Governance limit reached" not in result.message, breach
        assert "governance limit has been reached" not in result.system_prompt, breach


class RecordingClient:
    """Stands in for the backend client and keeps every payload the SDK would send."""
    def __init__(self):
        self.sent = []
        self.batches = []
        self.events_supported = self.heartbeat_supported = self.approvals_supported = True
        self.quarantine = False
        self.start_limits = None  # what run start answers with as the effective limits
        self.start_notices = None
        self.heartbeat_commands, self.checkpoint_commands = [], []  # handed out on the next call
        self.beats, self.approval_requests = [], []
        self.acks, self.lose_acks = [], False  # acks received; lose the next heartbeat that carries some

    def start_run(self, run_id, agent_name, *limits, context=None):
        if self.quarantine:
            return {"ok": False, "quarantined": True, "detail": "quarantined"}
        self.sent.append(("start", {"agent_name": agent_name, "context": context, "limits": limits}))
        resp = {"ok": True}
        if self.start_limits:
            resp["limits"] = self.start_limits
        if self.start_notices:
            resp["notices"] = self.start_notices
        return resp

    def checkpoint(self, *args, **kwargs):
        out, self.checkpoint_commands = self.checkpoint_commands, []
        return {"ok": True, "commands": out}

    def heartbeat(self, runs, acks=None):
        if not self.heartbeat_supported:
            return {"unsupported": True}
        if acks and self.lose_acks:
            self.lose_acks = False
            return {"network_error": "connection reset"}
        self.acks.extend(acks or [])
        self.beats.append(runs)
        out, self.heartbeat_commands = self.heartbeat_commands, []
        return {"commands": out}

    def request_approval(self, run_id, approval_id, reason, amount_usd, timeout_s):
        if not self.approvals_supported:
            return {"unsupported": True}
        self.approval_requests.append({"run_id": run_id, "approval_id": approval_id, "reason": reason,
                                       "amount_usd": amount_usd, "timeout_s": timeout_s})
        return self.approval_reply

    approval_reply = {"ok": True}  # set to {"ok": False, "status": 422} for a server that refuses the request

    def log_decision(self, run_id, iteration, reasoning, action):
        self.sent.append(("decision", {"reasoning": reasoning, "action": action}))

    def end_run(self, run_id, status, error=None, exception=None, breach=None):
        self.sent.append(("end", {"status": status, "error": error, "exception": exception, **({"breach": breach} if breach else {})}))

    def log_spans(self, run_id, spans):
        if not self.events_supported:
            return {"unsupported": True}
        self.batches.append(len(spans))
        self.sent.extend(("event", e) for e in spans)
        return {"ok": True}

    def of(self, kind):
        return [payload for k, payload in self.sent if k == kind]


def cloud(**init_args):
    """Cloud-mode init with the network replaced by a RecordingClient."""
    agentfences.init(api_key="fc_test", endpoint="http://fences.invalid", **init_args)
    agentfences.core._client = agentfences.control._client = recorder = RecordingClient()
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


def check_tool_calls():
    @agentfences.tool
    def web_search(query, limit=5):
        if query == "blocked":
            raise PermissionError("403 from search API")
        return [query] * limit

    @agentfences.tool(name="fetch")
    async def fetch_page(url):
        await asyncio.sleep(0.01)
        return "<html>"

    assert web_search("outside a run", limit=1) == ["outside a run"]  # no run: just calls through

    @governed(budget_usd=1)
    async def agent():
        web_search("cats", limit=2)
        await fetch_page("https://example.com/" + "x" * 500)
        try:
            web_search("blocked")
        except PermissionError:
            pass  # the tool's exception still reaches the agent
        return list(get_active_run().events)

    calls = asyncio.run(agent())
    a = [c["attributes"] for c in calls]
    assert [(c["name"], c["status"]) for c in calls] == [
        ("execute_tool web_search", "ok"), ("execute_tool fetch", "ok"), ("execute_tool web_search", "error")], calls
    assert a[0]["gen_ai.operation.name"] == "execute_tool" and a[0]["gen_ai.tool.name"] == "web_search"
    assert a[0]["gen_ai.tool.call.arguments"] == "query='cats', limit=2" and a[0]["fences.iteration"] == 0
    assert len(a[1]["gen_ai.tool.call.arguments"]) <= 300 and calls[1]["end_ts"] - calls[1]["start_ts"] >= 0.01
    assert a[2]["error.type"] == "PermissionError" and a[2]["fences.error.message"] == "PermissionError: 403 from search API"


def check_events_are_sent_batched_and_redacted():
    def scrub(e):
        if e["type"] != "span":
            return e
        if e["attributes"]["gen_ai.tool.name"] == "secret_tool":
            return None
        args = e["attributes"]["gen_ai.tool.call.arguments"].replace("hunter2", "***")
        return dict(e, attributes={**e["attributes"], "gen_ai.tool.call.arguments": args})
    sent = cloud(redact=scrub)
    try:
        @agentfences.tool
        def login(password):
            return True

        @agentfences.tool
        def secret_tool():
            return 1

        @governed(budget_usd=1)
        def agent():
            for _ in range(30):
                login("hunter2")
            secret_tool()
            log_decision("done logging in")
            return len(get_active_run().events)
        assert agent() == 31  # everything is kept on the run locally, even what's not sent
        agentfences.flush()
        tool_events = sent.of("event")
        assert len(tool_events) == 30 and all(e["attributes"]["gen_ai.tool.call.arguments"] == "password='***'" for e in tool_events)
        assert all("type" not in e for e in tool_events), "the redact-only type key isn't sent"
        assert sum(sent.batches) == 30 and len(sent.batches) < 30, sent.batches  # sent in batches, not one by one
        assert "hunter2" not in repr(sent.sent) and sent.of("decision")
    finally:
        agentfences.init(local_only=True)


def check_old_server_without_events():
    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    sent = cloud()
    sent.events_supported = False
    try:
        @agentfences.tool
        def ping():
            return "pong"

        @governed(budget_usd=1)
        def agent():
            for _ in range(5):
                ping()
                agentfences.flush()
            log_decision("still logged")
            return len(get_active_run().events)
        assert agent() == 5
        agentfences.flush()
        assert sent.of("decision"), "decisions still go through"
        assert len([m for m in seen if "doesn't accept tool and model call events" in m]) == 1, seen
    finally:
        agentfences.init(local_only=True)
        agentfences.outbox.spans_supported = True
        logging.getLogger("agentfences").removeHandler(handler)


def wait_for(condition, timeout=3.0):
    import time
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def background(fn):
    import threading
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def check_heartbeat_stop_and_limits():
    sent = cloud(heartbeat_s=0.05)
    try:
        @governed(budget_usd=1)
        async def agent():
            run = get_active_run()
            for step in range(1000):
                result = await checkpoint(cost_delta_usd=0.001)
                if result.breached:
                    return step, result, run.budget_usd
                await asyncio.sleep(0.02)

        def operator():
            wait_for(lambda: sent.beats)  # the heartbeat reports the live run...
            run_id = sent.beats[-1][0]["run_id"]
            assert set(sent.beats[-1][0]) == {"run_id", "iterations", "spent_usd", "tokens_used", "paused", "stopped"}
            sent.heartbeat_commands = [{"run_id": run_id, "type": "limits", "budget_usd": 5.0}]
            wait_for(lambda: not sent.heartbeat_commands)
            sent.heartbeat_commands = [{"run_id": run_id, "type": "stop"}]  # ...and brings back commands
        background(operator)
        steps, result, budget = asyncio.run(agent())
        assert result.breach_type == "stopped_by_user" and budget == 5.0, (result, budget)
        assert steps < 200, "stopped promptly by heartbeat, even though every checkpoint response said ok"
    finally:
        agentfences.init(local_only=True)


def check_pause_resume_and_timeout():
    import time
    sent = cloud(heartbeat_s=0.05)
    try:
        @governed(budget_usd=1)
        async def agent():
            started = time.monotonic()
            sent.checkpoint_commands = [{"type": "pause"}]  # a checkpoint response can carry commands too
            result = await checkpoint(cost_delta_usd=0.01)
            return result, time.monotonic() - started

        def operator():
            wait_for(lambda: sent.beats and sent.beats[-1] and sent.beats[-1][0]["paused"])  # heartbeat says it's paused
            time.sleep(0.3)
            sent.heartbeat_commands = [{"run_id": sent.beats[-1][0]["run_id"], "type": "resume"}]
        background(operator)
        result, waited = asyncio.run(agent())
        assert result.ok and waited >= 0.3, (result, waited)

        @governed(budget_usd=1)
        def sync_agent():
            sent.checkpoint_commands = [{"type": "pause"}]
            return checkpoint_sync(cost_delta_usd=0.01)
        background(operator)
        assert sync_agent().ok, "synchronous agents wait and resume too"

        agentfences.init(api_key="fc_test", endpoint="http://fences.invalid", heartbeat_s=0.05, pause_timeout_s=0.3)
        agentfences.core._client = sent = RecordingClient()
        agentfences.control._client = sent

        @governed(budget_usd=1)
        def forgotten():
            sent.checkpoint_commands = [{"type": "pause"}]
            return checkpoint_sync(cost_delta_usd=0.01)
        result = forgotten()  # paused and never resumed
        assert result.breach_type == "paused" and "not resumed in time" in result.message, result
    finally:
        agentfences.init(local_only=True)


def check_approvals():
    import time
    @governed(budget_usd=1)
    async def ask(timeout_s=600):
        approval = await agentfences.request_approval("Refund $240 to order 1182?", amount_usd=240, timeout_s=timeout_s)
        return approval, get_active_run().budget_usd

    approval, _ = asyncio.run(ask())  # local mode: no one to ask, so denied at once
    assert not approval.granted and "Local mode" in approval.note
    assert not asyncio.run(agentfences.request_approval("outside a run")).granted

    sent = cloud(heartbeat_s=0.05)
    try:
        def approver():
            wait_for(lambda: sent.approval_requests)
            req = sent.approval_requests[-1]
            assert req["reason"] == "Refund $240 to order 1182?" and req["amount_usd"] == 240 and req["timeout_s"] == 600
            time.sleep(0.1)
            sent.heartbeat_commands = [{"run_id": req["run_id"], "type": "approval", "approval_id": req["approval_id"],
                                        "granted": True, "by": "dana@example.com", "note": "ok, loyal customer", "amount_usd": 240}]
        background(approver)
        approval, budget = asyncio.run(ask())
        assert approval.granted and approval.by == "dana@example.com" and budget == 241, (approval, budget)
        agentfences.flush()
        span = [e for e in sent.of("event") if e["name"] == "fences.approval"][-1]  # request and answer as one span
        assert span["attributes"]["fences.approval.reason"] == "Refund $240 to order 1182?"
        assert span["attributes"]["fences.approval.granted"] is True and span["end_ts"] - span["start_ts"] >= 0.1

        @governed(budget_usd=1)
        def ask_sync():
            return agentfences.request_approval_sync("Delete the staging DB?", timeout_s=0.3)
        answer = ask_sync()  # nobody answers
        assert not answer.granted and "timeout" in answer.note

        sent.approval_reply = {"ok": False, "status": 422}  # refused: denied now, not after the 600 s timeout
        started = time.monotonic()
        approval, _ = asyncio.run(ask())
        assert not approval.granted and "HTTP 422" in approval.note and time.monotonic() - started < 1, approval
        assert not agentfences.approvals._asked
        sent.approval_reply = {"ok": True}

        sent.approvals_supported = False
        approval, _ = asyncio.run(ask())
        assert not approval.granted and "doesn't support approvals" in approval.note
    finally:
        agentfences.init(local_only=True)


def check_acks_and_containment_hooks():
    hooked = []

    def on_quarantine(event):
        hooked.append(event)

    def on_stop(event):
        hooked.append(event)
        raise RuntimeError("vault unreachable")

    sent = cloud(heartbeat_s=0.05, on_quarantine=on_quarantine, on_stop=on_stop)
    sent.lose_acks = True  # the first heartbeat with acks is lost: they must go with the next
    try:
        @governed(budget_usd=10)
        async def agent():
            for _ in range(400):
                result = await checkpoint(cost_delta_usd=0.001)
                if result.breached:
                    return result.breach_type
                await asyncio.sleep(0.01)

        def operator():
            wait_for(lambda: len(sent.beats) >= 1 and len(sent.beats[-1]) == 2)
            ids = [r["run_id"] for r in sent.beats[-1]]
            quarantine = [{"run_id": i, "type": "quarantine", "id": "q1", "by": "dana@example.com", "reason": "leaking data"} for i in ids]
            sent.heartbeat_commands = quarantine
            wait_for(lambda: all(r["paused"] for r in sent.beats[-1]))  # both runs hold at their checkpoints
            sent.heartbeat_commands = quarantine  # re-sent with the same id: applied again, not acknowledged again
            wait_for(lambda: not sent.heartbeat_commands)
            sent.heartbeat_commands = [{"run_id": i, "type": "resume", "id": f"r-{i}"} for i in ids]  # quarantine released
            wait_for(lambda: not any(r["paused"] for r in sent.beats[-1]))
            sent.heartbeat_commands = [{"run_id": ids[0], "type": "stop", "id": "s1"}, {"run_id": ids[1], "type": "stop"}]
        background(operator)

        async def both():
            return await asyncio.gather(agent(), agent())
        assert asyncio.run(both()) == ["stopped_by_user"] * 2

        wait_for(lambda: any(a.get("hook") == "failed" for a in sent.acks))
        quarantines = [e for e in hooked if e["type"] == "quarantine"]
        assert len(quarantines) == 1 and quarantines[0]["reason"] == "leaking data", hooked  # once per process per quarantine
        assert len([e for e in hooked if e["type"] == "stop"]) == 2, "on_stop runs once per run, even without a command id"
        receipts = [(a["id"], a["type"]) for a in sent.acks if "hook" not in a]
        assert sorted(receipts) == sorted([("q1", "quarantine")] * 2 + [(f"r-{i}", "resume") for i in {a["run_id"] for a in sent.acks if a["id"] == "q1"}]
                                          + [("s1", "stop")]), receipts
        hooks = {a["id"]: a for a in sent.acks if "hook" in a}
        assert hooks["q1"]["hook"] == "ok" and hooks["s1"]["hook"] == "failed" and "vault unreachable" in hooks["s1"]["detail"], hooks
        assert all("received_at" in a for a in sent.acks if "hook" not in a)
    finally:
        agentfences.init(local_only=True)


def check_hooks_finish_before_exit():
    """A script that ends right after a stop still finishes on_stop, and acknowledges it, before the process exits."""
    import json, os, subprocess, sys, tempfile
    out = os.path.join(tempfile.mkdtemp(), "revoked")
    script = f"""
import atexit, json, sys, time
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
acks = []
atexit.register(lambda: print(json.dumps(acks)))  # registered first, so it runs after Fences' own exit handlers
import agentfences
from test_local import RecordingClient

def on_stop(event):
    time.sleep(0.5)  # revoking a credential takes a moment
    open({out!r}, "w").write("revoked")

agentfences.init(api_key="fc", endpoint="http://x.invalid", heartbeat_s=60, on_stop=on_stop)
sent = RecordingClient()
agentfences.core._client = agentfences.control._client = sent
acks = sent.acks
sent.checkpoint_commands = [{{"type": "stop", "id": "s1"}}]

@agentfences.governed(budget_usd=1)
def agent():
    return agentfences.checkpoint_sync(cost_delta_usd=0.01).breach_type
assert agent() == "stopped_by_user"
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert open(out).read() == "revoked", "the hook was killed with the process"
    acks = json.loads(r.stdout.strip().splitlines()[-1])
    assert [(a["id"], a.get("hook")) for a in acks] == [("s1", None), ("s1", "ok")], acks


def check_acks_survive_a_rate_limited_heartbeat():
    """A heartbeat over the server's heartbeat budget gets 429; its acks go again with the next one."""
    sent = cloud(heartbeat_s=0.1)
    real = sent.heartbeat
    limited = []

    def heartbeat(runs, acks=None):
        if acks and not limited:
            limited.append(acks)
            return {"ok": False, "status": 429}
        return real(runs, acks)
    sent.heartbeat = heartbeat
    sent.checkpoint_commands = [{"type": "stop", "id": "s1"}]
    try:
        @governed(budget_usd=1)
        def agent():
            checkpoint_sync(cost_delta_usd=0.01)
            wait_for(lambda: sent.acks)
        agent()
        assert limited and [a["id"] for a in sent.acks] == ["s1"], (limited, sent.acks)
    finally:
        agentfences.init(local_only=True)


def check_resent_quarantine_doesnt_flood_heartbeats():
    """The server re-sends quarantine on every reply while it lasts; that mustn't make heartbeats back-to-back."""
    import time
    sent = cloud(heartbeat_s=0.5, pause_timeout_s=1.5)
    real = sent.heartbeat

    def quarantined(runs, acks=None):
        real(runs, acks)
        return {"commands": [{"run_id": r["run_id"], "type": "quarantine", "id": "q1"} for r in runs]}
    sent.heartbeat = quarantined
    try:
        @governed(budget_usd=1)
        def agent():
            time.sleep(0.6)  # the first heartbeat brings the quarantine
            before, t = len(sent.beats), time.monotonic()
            checkpoint_sync(cost_delta_usd=0.01)  # held until pause_timeout_s
            return len(sent.beats) - before, time.monotonic() - t
        beats, held = agent()
        assert beats <= held / 0.5 + 3, f"{beats} heartbeats in {held:.1f}s: the re-sent quarantine floods the server"
    finally:
        agentfences.init(local_only=True)


def check_quarantine_blocks_guarded_tools():
    cloud(heartbeat_s=60)
    try:
        @agentfences.tool(guard=True)
        def deploy():
            raise AssertionError("ran while quarantined")

        @governed(budget_usd=1)
        def agent():
            agentfences.control.apply(get_active_run(), [{"type": "quarantine", "id": "q2"}])
            try:
                deploy()
            except agentfences.ToolBlocked as e:
                return e.reason
        assert "quarantined" in agent()  # at once, not after the quarantine's pause is released
    finally:
        agentfences.init(local_only=True)


def check_server_limits_are_adopted():
    sent = cloud()
    sent.start_limits = {"budget_usd": 0.05, "max_iterations": 3, "max_duration_ms": 300000, "max_tokens": 0}
    try:
        @governed(budget_usd=10, max_iterations=100)  # the code asks for more than the agent's ceilings allow
        def agent():
            run = get_active_run()
            return run.budget_usd, run.max_iterations
        assert agent() == (0.05, 3)  # so the offline fallback enforces the server's capped numbers
    finally:
        agentfences.init(local_only=True)


def warnings_during(fn):
    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    try:
        return fn(), seen
    finally:
        logging.getLogger("agentfences").removeHandler(handler)


def check_no_budget_runs_but_warns():
    agentfences.core._noticed.clear()

    @governed()  # no limits in code at all
    def unlimited_agent():
        run = get_active_run()
        for _ in range(50):
            assert checkpoint_sync(cost_delta_usd=1000).ok  # spend isn't capped...
        return run.budget_usd, run.max_iterations, run.max_duration_ms, run.max_tokens

    (limits, seen) = warnings_during(lambda: [unlimited_agent(), unlimited_agent()])
    assert limits[0] == (None, 100, 300_000, 0), limits  # ...the other limits take their defaults
    nudges = [m for m in seen if "unlimited_agent has no budget" in m]
    assert len(nudges) == 1 and "@governed(budget_usd=...)" in nudges[0], seen  # once per agent, not per run

    @governed()
    def looping_agent():
        for _ in range(1000):
            r = checkpoint_sync(cost_delta_usd=1)
            if r.breached:
                return r
    r = looping_agent()  # with no budget, the step limit still stops it, and its message still formats
    assert r.breach_type == "iteration_limit" and "101 steps" in r.message, r


def check_dashboard_owned_limits():
    agentfences.core._noticed.clear()
    sent = cloud()
    try:
        sent.start_limits = {"budget_usd": 2.5, "max_iterations": 40, "max_duration_ms": 60_000, "max_tokens": 0}

        @governed(max_iterations=50)  # only some limits in code: the rest come from the dashboard
        def owned():
            r = get_active_run()
            return r.budget_usd, r.max_iterations
        assert owned() == (2.5, 40)
        assert sent.of("start")[-1]["limits"] == (None, 50, None, None), "unset limits go as null"

        sent.start_limits = {"budget_usd": None, "max_iterations": 100, "max_duration_ms": 300_000, "max_tokens": 0}
        sent.start_notices = [{"code": "no_budget", "message": "free_agent has no budget, so its spend isn't capped.",
                               "url": "https://fences.example/agents/free_agent"}]

        @governed()
        def free_agent():
            return get_active_run().budget_usd
        result, seen = warnings_during(lambda: [free_agent(), free_agent()])
        assert result == [None, None]
        assert seen.count("agentfences: free_agent has no budget, so its spend isn't capped. https://fences.example/agents/free_agent") == 1, seen

        sent.start_limits = sent.start_notices = None  # an older server: no limits in the reply

        @governed()
        def old_server_agent():
            return get_active_run().budget_usd
        result, seen = warnings_during(old_server_agent)
        assert result is None and any("old_server_agent has no budget" in m and "dashboard" in m for m in seen), seen
    finally:
        agentfences.init(local_only=True)


def check_version_is_sent():
    from importlib.metadata import version
    from agentfences.client import GovClient
    assert version("agentfences") == agentfences.__version__  # the package build reads the one version
    headers = GovClient("fc_x", "http://fences.invalid").headers
    assert headers["X-Fences-SDK"] == f"python/{agentfences.__version__}" and headers["X-API-Key"] == "fc_x"


def check_quarantine():
    sent = cloud()
    sent.quarantine = True
    ran = []
    try:
        @governed(budget_usd=1)
        def agent():
            ran.append(True)
        try:
            agent()
        except agentfences.AgentQuarantined as e:
            assert e.agent_name == "agent" and "quarantined" in str(e)
        else:
            raise AssertionError("a quarantined agent must not run")
        assert not ran, "no agent code runs"
    finally:
        agentfences.init(local_only=True)


def check_bad_amounts_are_refused():
    """A NaN, negative or non-number cost would switch the budget off: refused loudly instead."""
    @agentfences.tool(cost_usd=lambda amount: amount)
    def pay(amount):
        ran.append(amount)

    @governed(budget_usd=0.01)
    def agent(**step):
        checkpoint_sync(**step)

    for step, error in (({"cost_delta_usd": float("nan")}, TypeError), ({"cost_delta_usd": -100}, ValueError),
                        ({"cost_delta_usd": float("inf")}, ValueError), ({"cost_delta_usd": "0.02"}, TypeError),
                        ({"cost_delta_usd": True}, TypeError), ({"tokens_used": -5}, ValueError)):
        try:
            agent(**step)
        except error as e:
            assert "cost_delta_usd" in str(e) or "tokens_used" in str(e), e
        else:
            raise AssertionError(f"checkpoint({step}) should raise {error.__name__}")

    ran = []

    @governed(budget_usd=1)
    def tools(amount):
        try:
            pay(amount)
        except (TypeError, ValueError) as e:
            return type(e).__name__, get_active_run().reserved_usd, get_active_run().cost_usd
    assert tools(-50) == ("ValueError", 0.0, 0.0) and tools(float("nan")) == ("TypeError", 0.0, 0.0) and not ran

    from decimal import Decimal  # money often is: accepted, as a float

    @governed(budget_usd=1)
    def decimal_amounts():
        pay(Decimal("0.25"))
        checkpoint_sync(cost_delta_usd=Decimal("0.05"))
        return get_active_run().cost_usd, get_active_run().reserved_usd
    assert decimal_amounts() == (0.3, 0.0) and ran == [Decimal("0.25")]


def check_custom_prices_are_validated():
    for prices in ({"m": {"input": 1.0}}, {"m": {"output": 1.0}}, {"m": {"input": -1, "output": 1}},
                   {"m": {"input": float("nan"), "output": 1}}, {"m": {"input": "1", "output": 1}}, {"m": 3.0}):
        try:
            agentfences.init(local_only=True, prices=prices)
        except ValueError as e:
            assert "'m'" in str(e), e
        else:
            raise AssertionError(f"init(prices={prices}) should raise ValueError")
    agentfences.init(local_only=True, prices={"m": {"input": 1.0, "output": 2.0}})
    assert price_of("m") == {"in": 1e-6, "out": 2e-6}
    agentfences.init(local_only=True)


def check_client_replies():
    """GovClient never raises mid-run: a revoked key or an odd reply comes back as a value the SDK handles."""
    from agentfences.client import GovClient

    class Reply:
        def __init__(self, status, body):
            self.status_code, self.ok, self._body = status, status < 400, body
            self.text = body if isinstance(body, str) else str(body)

        def json(self):
            if isinstance(self._body, str):
                raise ValueError("not JSON")
            return self._body

    client, replies = GovClient("fc_test", "http://fences.invalid"), []
    client._session = lambda: NS(post=lambda *a, **k: replies.pop(0))

    def call(status, body, fn):
        replies.append(Reply(status, body))
        return fn()
    start = lambda: client.start_run("r", "a", 1, None, None, None)
    try:
        call(401, {"detail": "Invalid or revoked API key"}, start)
    except agentfences.FencesAuthError as e:
        assert isinstance(e, PermissionError) and "revoked" in str(e)
    else:
        raise AssertionError("a rejected key at run start should raise FencesAuthError")
    assert call(403, {"detail": "no"}, lambda: client.checkpoint("r", 0, 1, 0, 0)) == {
        "network_error": "API key rejected (HTTP 403)", "auth_rejected": True}
    assert call(401, {"detail": "no"}, lambda: client.end_run("r", "success"))["auth_rejected"]
    assert call(200, ["not", "a", "dict"], lambda: client.checkpoint("r", 0, 1, 0, 0)) == {"network_error": "response was not JSON"}
    assert call(200, "<html>", lambda: client.checkpoint("r", 0, 1, 0, 0)) == {"network_error": "response was not JSON"}
    assert call(404, {"detail": "Not Found"}, lambda: client.guard("r", {})) == {"unsupported": True}
    assert call(401, {"detail": "no"}, lambda: client.guard("r", {})) == {"ok": False, "status": 401}
    assert call(200, [1], lambda: client.guard("r", {})) == {}  # not a reply: the guard then fails closed


def check_revoked_key_and_odd_replies():
    """A key revoked mid-run ends the run at its next checkpoint, and the agent's result still comes back.
    Null or odd numbers in a reply never break local enforcement later."""
    sent = cloud()
    try:
        replies = []
        sent.checkpoint = lambda *a, **k: replies.pop(0)

        @governed(budget_usd=1)
        def agent():
            return [checkpoint_sync(cost_delta_usd=0.1) for _ in range(len(replies))]

        replies[:] = [{"ok": True, "spent_usd": None, "iterations": "3", "max_iterations": None, "budget_usd": None},
                      {"network_error": "HTTP 500"}]
        first, second = agent()
        assert first.ok and second.ok, (first, second)  # unlimited (null budget), 2 steps of 100, no TypeError
        end = sent.of("end")[-1]
        assert end["status"] == "success", end

        replies[:] = [{"ok": True}, {"network_error": "API key rejected (HTTP 401)", "auth_rejected": True}]
        first, second = agent()
        assert first.ok and second.breach_type == "fences_unreachable" and sent.of("end")[-1]["status"] == "breached"
    finally:
        agentfences.init(local_only=True)


def check_circuit_breaker():
    """After 3 failed calls in a row, calls are skipped (no waiting on timeouts) until one test call gets through."""
    import requests
    from agentfences.client import GovClient, FAILURES_TO_OPEN

    client, tried, replies = GovClient("fc_test", "http://fences.invalid"), [], []

    def post(*a, **k):
        tried.append(1)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return NS(status_code=r, ok=r < 400, text="", json=lambda: {"ok": True})
    client._session = lambda: NS(post=post)
    call = lambda: client.checkpoint("r", 0, 1, 0, 0)

    replies[:] = [requests.ConnectionError("refused"), 503, 429]  # each kind of failure counts
    assert all("network_error" in call() for _ in range(FAILURES_TO_OPEN)) and len(tried) == 3
    assert call()["network_error"].startswith("Fences unreachable") and len(tried) == 3  # skipped: open

    client.breaker.open_until = 0  # the pause is over: one test call goes through, and fails again
    replies[:] = [requests.Timeout("slow")]
    assert "network_error" in call() and len(tried) == 4
    assert "network_error" in call() and len(tried) == 4  # open again

    client.breaker.open_until = 0  # the server is back: closed, calls go through again
    replies[:] = [200, 200]
    assert call() == {"ok": True} and call() == {"ok": True} and len(tried) == 6

    replies[:] = [400, 422, 401, 200]  # the server answering, even with an error, isn't a failure
    for _ in range(4):
        call()
    assert client.breaker.failures == 0

    replies[:] = [429] * 5  # heartbeats over their own budget don't cut off checkpoints
    for _ in range(5):
        client.heartbeat([])
    assert client.breaker.failures == 0 and len(tried) == 15


def check_step_ids_and_one_retry():
    """Each checkpoint carries its own step_id; once the server shows it counts a step once (by echoing it),
    a checkpoint that hits a connection error goes once more. Guards always may. Timeouts aren't retried."""
    import requests
    from agentfences.client import GovClient

    client, posts, replies = GovClient("fc_test", "http://fences.invalid"), [], []

    def post(url, json=None, **k):
        posts.append((url.rsplit("/", 1)[-1], json))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return NS(status_code=200, ok=True, text="", json=lambda: r)
    client._session = lambda: NS(post=post)
    step = lambda sid: client.checkpoint("r", 0.01, 1, 0, 0, step_id=sid)

    replies[:] = [requests.ConnectionError("reset")]  # the server hasn't shown it dedupes: no retry
    assert "network_error" in step("s1") and len(posts) == 1 and posts[0][1]["step_id"] == "s1"
    replies[:] = [{"ok": True, "step_id": "s2"}]       # now it has
    assert step("s2")["ok"] and client.dedupes_steps
    posts.clear()
    replies[:] = [requests.ConnectionError("reset"), {"ok": True, "step_id": "s3"}]
    assert step("s3")["ok"] and [p[1]["step_id"] for p in posts] == ["s3", "s3"]  # the same step, twice
    posts.clear()
    replies[:] = [requests.ReadTimeout("slow")]
    assert "network_error" in step("s4") and len(posts) == 1  # a timeout isn't retried
    posts.clear()
    replies[:] = [requests.ConnectTimeout("slow connect")]  # a ConnectionError too, but a timeout
    assert "network_error" in step("s5") and len(posts) == 1
    posts.clear()
    replies[:] = [requests.ConnectionError("reset"), {"allow": True}]
    assert client.guard("r", {"guard_id": "g1", "phase": "check"}) == {"allow": True} and len(posts) == 2

    # The SDK gives every step its own id
    sent = cloud()
    try:
        ids = []
        sent.checkpoint = lambda *a, step_id=None, **k: ids.append(step_id) or {"ok": True}

        @governed(budget_usd=1)
        def agent():
            checkpoint_sync(cost_delta_usd=0.01)
            checkpoint_sync(cost_delta_usd=0.01)
        agent()
        assert len(ids) == 2 and len(set(ids)) == 2 and all(ids), ids
    finally:
        agentfences.init(local_only=True)


def check_old_server_without_control():
    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    sent = cloud(heartbeat_s=0.05)
    sent.heartbeat_supported = False
    try:
        @governed(budget_usd=1)
        async def agent():
            for _ in range(10):
                await asyncio.sleep(0.03)
                assert (await checkpoint(cost_delta_usd=0.01)).ok
            return "done"
        assert asyncio.run(agent()) == "done"
        assert len([m for m in seen if "doesn't support the heartbeat" in m]) == 1, seen
    finally:
        agentfences.init(local_only=True)
        logging.getLogger("agentfences").removeHandler(handler)


if __name__ == "__main__":
    check_requires_init()
    check_breach_messages()
    agentfences.init(local_only=True)
    checks = [
        check_readme_quickstart, check_limits_trip_once_exceeded, check_time_limit,
        check_removed_apis_fail_loudly, check_sync_agent, check_streaming_agent,
        check_concurrent_runs_are_independent, check_nested_runs, check_outside_a_run,
        check_prices_each_response_shape, check_extra_costs_and_old_calls, check_budget_stops_on_real_usage,
        check_unknown_model_warns_and_custom_prices, check_run_context, check_exception_capture, check_redaction,
        check_tool_calls, check_events_are_sent_batched_and_redacted, check_old_server_without_events,
        check_heartbeat_stop_and_limits, check_pause_resume_and_timeout, check_approvals,
        check_acks_and_containment_hooks, check_quarantine_blocks_guarded_tools, check_hooks_finish_before_exit,
        check_resent_quarantine_doesnt_flood_heartbeats, check_acks_survive_a_rate_limited_heartbeat, check_server_limits_are_adopted, check_no_budget_runs_but_warns, check_dashboard_owned_limits,
        check_version_is_sent, check_quarantine,
        check_old_server_without_control, check_bad_amounts_are_refused, check_custom_prices_are_validated,
        check_client_replies, check_revoked_key_and_odd_replies, check_circuit_breaker,
        check_step_ids_and_one_retry,
    ]
    for check in checks:
        try:
            check()
        except BaseException:
            if os.environ.get("GITHUB_ACTIONS"):  # an annotation: readable without the job's log
                import traceback
                print(f"::error title={check.__name__}::" + traceback.format_exc()[-1500:].replace("\n", "%0A"))
            raise
        print("ok", check.__name__)
    agentfences.flush()
    print("all local checks passed")
