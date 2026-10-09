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
        self.batches = []
        self.events_supported = self.heartbeat_supported = self.approvals_supported = True
        self.quarantine = False
        self.start_limits = None  # what run start answers with as the effective limits
        self.start_notices = None
        self.heartbeat_commands, self.checkpoint_commands = [], []  # handed out on the next call
        self.beats, self.approval_requests = [], []

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

    def checkpoint(self, *args):
        out, self.checkpoint_commands = self.checkpoint_commands, []
        return {"ok": True, "commands": out}

    def heartbeat(self, runs):
        if not self.heartbeat_supported:
            return {"unsupported": True}
        self.beats.append(runs)
        out, self.heartbeat_commands = self.heartbeat_commands, []
        return {"commands": out}

    def request_approval(self, run_id, approval_id, reason, amount_usd):
        if not self.approvals_supported:
            return {"unsupported": True}
        self.approval_requests.append({"run_id": run_id, "approval_id": approval_id, "reason": reason, "amount_usd": amount_usd})
        return {"ok": True}

    def log_decision(self, run_id, iteration, reasoning, action):
        self.sent.append(("decision", {"reasoning": reasoning, "action": action}))

    def end_run(self, run_id, status, error=None, exception=None):
        self.sent.append(("end", {"status": status, "error": error, "exception": exception}))

    def log_events(self, run_id, events):
        if not self.events_supported:
            return {"unsupported": True}
        self.batches.append(len(events))
        self.sent.extend(("event", e) for e in events)
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
    assert [(c["name"], c["ok"]) for c in calls] == [("web_search", True), ("fetch", True), ("web_search", False)], calls
    assert calls[0]["args"] == "query='cats', limit=2" and calls[0]["type"] == "tool_call"
    assert len(calls[1]["args"]) <= 300 and calls[1]["latency_ms"] >= 10
    assert calls[2]["error"] == "PermissionError: 403 from search API"


def check_events_are_sent_batched_and_redacted():
    sent = cloud(redact=lambda e: None if e.get("name") == "secret_tool" else
                 dict(e, args=e["args"].replace("hunter2", "***")) if e["type"] == "tool_call" else e)
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
        assert len(tool_events) == 30 and all(e["args"] == "password='***'" for e in tool_events)
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
        agentfences.core._events_supported = True
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
            assert set(sent.beats[-1][0]) == {"run_id", "iterations", "spent_usd", "tokens_used", "paused"}
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
            assert req["reason"] == "Refund $240 to order 1182?" and req["amount_usd"] == 240
            time.sleep(0.1)
            sent.heartbeat_commands = [{"run_id": req["run_id"], "type": "approval", "approval_id": req["approval_id"],
                                        "granted": True, "by": "dana@example.com", "note": "ok, loyal customer", "amount_usd": 240}]
        background(approver)
        approval, budget = asyncio.run(ask())
        assert approval.granted and approval.by == "dana@example.com" and budget == 241, (approval, budget)

        @governed(budget_usd=1)
        def ask_sync():
            return agentfences.request_approval_sync("Delete the staging DB?", timeout_s=0.3)
        answer = ask_sync()  # nobody answers
        assert not answer.granted and "timeout" in answer.note

        sent.approvals_supported = False
        approval, _ = asyncio.run(ask())
        assert not approval.granted and "doesn't support approvals" in approval.note
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
        check_heartbeat_stop_and_limits, check_pause_resume_and_timeout, check_approvals, check_server_limits_are_adopted, check_no_budget_runs_but_warns, check_dashboard_owned_limits,
        check_quarantine,
        check_old_server_without_control,
    ]
    for check in checks:
        check()
        print("ok", check.__name__)
    agentfences.flush()
    print("all local checks passed")
