"""
The checks behind ENFORCEMENT.md: what Fences still guarantees when things go wrong.
Parallel calls racing for one budget, an unreachable server, a stop while a tool hangs.
No network: the server is a stand-in.

    pip install ./sdk && python tests/test_enforcement.py
"""
import asyncio
import contextvars
import logging
import math
import threading
import time

import agentfences
from agentfences import governed, checkpoint, checkpoint_sync, get_active_run, tool, ToolBlocked, FencesStop
from test_local import RecordingClient, wait_for, background  # also blocks the network


class Server(RecordingClient):
    """The stand-in server, with guards: allows a call while spent + held + amount fits the budget."""
    def __init__(self, budget=1.0):
        super().__init__()
        self.budget, self.spent, self.held = budget, 0.0, {}
        self.guards, self.guard_supported, self.reachable = [], True, True
        self.needs_approval = False
        self.lock = threading.Lock()

    breach = None  # set to make every checkpoint reply breached, like a run over its limit

    def checkpoint(self, *args):
        if not self.reachable:
            return {"network_error": "connection refused"}
        if self.breach:
            return {"ok": False, "breach": self.breach}
        return super().checkpoint(*args)

    def guard(self, run_id, body):
        if not self.reachable:
            return {"network_error": "connection refused"}
        if not self.guard_supported:
            return {"unsupported": True}
        self.guards.append(body)
        with self.lock:
            if body["phase"] == "settle":
                self.held.pop(body["guard_id"], None)
                self.spent += body["cost_usd"]
                return {"ok": True}
            if self.needs_approval and not body["approval_id"]:
                return {"allow": False, "needs_approval": True, "reason": "issue_refund needs approval"}
            if self.spent + sum(self.held.values()) + body["amount_usd"] > self.budget:
                return {"allow": False, "reason": "over budget", "breach": "budget_exceeded"}
            self.held[body["guard_id"]] = body["amount_usd"]
            return {"allow": True}


def cloud(server, **init_args):
    agentfences.init(api_key="fc_test", endpoint="http://fences.invalid", heartbeat_s=0.05, **init_args)
    agentfences.core._client = agentfences.control._client = server
    return server


def blocked(fn):
    """The ToolBlocked fn raises, or None if it ran."""
    try:
        fn()
    except ToolBlocked as e:
        return e
    return None


# ── Level 3: guarded tools ────────────────────────────────────────────────────

def check_parallel_calls_cannot_overspend():
    """Five $0.40 refunds at once against a $1 budget: exactly two run, in local mode and against a server."""
    ran = []

    @tool(cost_usd=lambda amount: amount)
    async def refund(amount):
        await asyncio.sleep(0.05)  # all five are in flight together
        ran.append(amount)

    @governed(budget_usd=1)
    async def agent():
        results = await asyncio.gather(*(refund(0.4) for _ in range(5)), return_exceptions=True)
        run = get_active_run()
        return [type(r).__name__ for r in results], run.cost_usd, run.reserved_usd

    names, spent, held = asyncio.run(agent())
    assert sorted(names) == ["NoneType"] * 2 + ["ToolBlocked"] * 3 and len(ran) == 2, names
    assert round(spent, 9) == 0.8 and held == 0, (spent, held)

    @tool(cost_usd=0.4)
    def sync_refund():
        time.sleep(0.05)

    @governed(budget_usd=1)
    def threaded():  # threads racing in one run
        outcomes = []
        ts = [threading.Thread(target=contextvars.copy_context().run, args=(lambda: outcomes.append(blocked(sync_refund)),))
              for _ in range(5)]
        for t in ts: t.start()
        for t in ts: t.join()
        return sum(o is None for o in outcomes)
    assert threaded() == 2

    server = cloud(Server(budget=1.0))
    try:
        ran.clear()
        names, _, _ = asyncio.run(agent())
        assert names.count("ToolBlocked") == 3 and len(ran) == 2
        assert round(server.spent, 9) == 0.8 and not server.held, (server.spent, server.held)
        checks = [g for g in server.guards if g["phase"] == "check"]
        assert {"guard_id", "tool", "arguments", "amount_usd", "approval_id"} <= set(checks[0]) and checks[0]["tool"] == "refund"
    finally:
        agentfences.init(local_only=True)


def check_blocked_tools_never_run():
    ran = []

    @tool(guard=True)
    def delete_records():
        ran.append(1)

    @governed(budget_usd=0.05)
    def over_budget():
        checkpoint_sync(cost_delta_usd=0.1)
        return blocked(delete_records)
    e = over_budget()
    assert e and e.breach_type == "budget_exceeded" and not ran, e
    assert blocked(delete_records) is None and ran, "outside a governed run a guarded tool just runs"

    @tool(approve=True)
    def wire_money():
        ran.append("wire")

    @governed(budget_usd=1)
    def local_approval():
        return blocked(wire_money)
    assert "Local mode" in local_approval().reason and "wire" not in ran  # no one to approve locally


def check_failed_and_timed_out_calls_are_settled():
    @tool(cost_usd=0.3)
    def declined():
        raise ValueError("card declined")

    @tool(cost_usd=0.3, timeout_s=0.05)
    async def slow_payment():
        await asyncio.sleep(5)

    @governed(budget_usd=1)
    async def agent():
        run = get_active_run()
        try:
            declined()
        except ValueError:
            pass
        assert (run.cost_usd, run.reserved_usd) == (0, 0), "an error releases the reservation, uncharged"
        try:
            await slow_payment()
        except TimeoutError:
            pass
        return run.cost_usd, run.reserved_usd
    assert asyncio.run(agent()) == (0.3, 0), "a timeout may have paid, so it's charged"


def check_guard_fails_closed():
    ran = []

    @tool(cost_usd=0.1)
    def charge_card():
        ran.append(1)

    @governed(budget_usd=1)
    def agent():
        return blocked(charge_card)

    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logging.getLogger("agentfences").addHandler(handler)
    server = cloud(Server())
    try:
        server.reachable = False
        assert "couldn't reach Fences" in agent().reason
        server.reachable, server.guard_supported = True, False
        assert "doesn't support guarded tools" in agent().reason and "doesn't support guarded tools" in agent().reason
        assert len([m for m in seen if "guarded tools" in m]) == 1, seen
        assert not ran
    finally:
        agentfences.init(local_only=True)
        logging.getLogger("agentfences").removeHandler(handler)


def check_server_required_approval():
    ran = []

    @tool(cost_usd=lambda order_id, amount: amount)
    def issue_refund(order_id, amount):
        ran.append(order_id)

    @governed(budget_usd=1)
    def agent():
        issue_refund("1182", 0.5)

    server = cloud(Server())
    server.needs_approval = True
    try:
        def approver():
            wait_for(lambda: server.approval_requests)
            req = server.approval_requests[-1]
            assert req["reason"] == "issue_refund(order_id='1182', amount=0.5)" and req["amount_usd"] == 0.5, req
            server.heartbeat_commands = [{"run_id": req["run_id"], "type": "approval", "approval_id": req["approval_id"],
                                          "granted": True, "by": "dana@example.com"}]
        background(approver)
        agent()
        checks = [g for g in server.guards if g["phase"] == "check"]
        assert ran == ["1182"] and len(checks) == 2 and checks[1]["approval_id"] == server.approval_requests[-1]["approval_id"]
    finally:
        agentfences.init(local_only=True)


def check_mcp_session_calls_are_recorded_and_guarded():
    """agentfences.mcp.wrap: every call_tool is a tool span; named tools are guarded; isError is a failed call."""
    from types import SimpleNamespace as NS

    class Session:  # stands in for mcp.ClientSession
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
            self.calls.append(name)
            return NS(isError=name == "broken", content=[NS(type="text", text="disk full")])

    session = agentfences.mcp.wrap(Session(), guard="send_email", cost_usd={"refund": lambda a: a["amount"]})
    assert asyncio.run(session.call_tool("search", {"q": "x"})).isError is False  # outside a run: straight through

    @governed(budget_usd=1)
    async def agent():
        await session.call_tool("search", {"q": "cats"})
        failed = await session.call_tool("broken", {})
        await session.call_tool("send_email", {"to": "a@b.c"})
        await session.call_tool("refund", {"amount": 0.7})
        denied = await asyncio.gather(session.call_tool("refund", {"amount": 0.7}), return_exceptions=True)
        return failed, denied[0], list(get_active_run().events)

    server = cloud(Server())
    try:
        failed, denied, spans = asyncio.run(agent())
    finally:
        agentfences.init(local_only=True)
    assert failed.isError and isinstance(denied, ToolBlocked) and denied.breach_type == "budget_exceeded", denied
    assert session.calls == ["search", "search", "broken", "send_email", "refund"], session.calls  # the refused refund never ran
    a = [e["attributes"] for e in spans]
    assert [(x["gen_ai.tool.name"], e["status"]) for x, e in zip(a, spans)] == [
        ("search", "ok"), ("broken", "error"), ("send_email", "ok"), ("refund", "ok"), ("refund", "error")], spans
    assert a[0]["gen_ai.tool.call.arguments"] == "q='cats'" and a[1]["error.type"] == "MCPToolError"
    checks = [(g["tool"], g["amount_usd"]) for g in server.guards if g["phase"] == "check"]
    assert checks == [("send_email", 0.0), ("refund", 0.7)], checks  # the second refund is refused here, over what the run already spent
    assert round(server.spent, 9) == 0.7 and not server.held


# ── Level 1: model calls are prechecked against the budget ───────────────────

def check_model_call_worst_case():
    from agentfences.precheck import worst_case_usd
    from agentfences.pricing import price_of
    p = price_of("gpt-4o")
    assert math.isclose(worst_case_usd("gpt-4o", "x" * 400, 1000), 201 * p["in"] + 1000 * p["out"])  # 402 JSON bytes / 2
    assert math.isclose(worst_case_usd("gpt-4o", "x" * 400), 201 * p["in"] + p["max_out"] * p["out"]), "no cap: the model's"
    image = "data:image/png;base64," + "A" * 100_000  # a file counts as a flat 2,000 tokens, not its base64 size
    assert math.isclose(worst_case_usd("gpt-4o", image, 1000), (1 + 2000) * p["in"] + 1000 * p["out"])
    assert worst_case_usd("gpt-4o", "hi", 1000, n=3) > 3000 * p["out"], "n choices multiply the output"
    assert worst_case_usd("no-such-model", "hi", 1000) is None


def check_parallel_model_calls_cannot_overspend():
    """Ten calls in flight at once, each worth up to ~$0.01, against $0.05: only those that fit go out."""
    from agentfences import precheck

    @governed(budget_usd=0.05)
    async def agent():
        run = get_active_run()

        async def call():
            h = precheck.hold("gpt-4o", "hi", {"max_tokens": 1000})  # raises FencesStop if it might not fit
            await asyncio.sleep(0.05)  # all in flight together
            precheck.finish(h)
            return h[0]
        results = await asyncio.gather(*(call() for _ in range(10)), return_exceptions=True)
        made = [r for r in results if isinstance(r, float)]
        assert run.cost_usd + run.reserved_usd <= 0.05
        await checkpoint(cost_delta_usd=sum(made))  # records what they cost, and releases their holds
        return len(made), sum(isinstance(r, FencesStop) for r in results), run.reserved_usd, run.cost_usd

    made, refused, held, spent = asyncio.run(agent())
    assert (made, refused) == (4, 6) and held == 0 and spent <= 0.05, (made, refused, held, spent)


def check_holds_are_released_exactly():
    """Equal holds from parallel calls: releasing one never releases another."""
    from agentfences import precheck

    @governed(budget_usd=1)
    def agent():
        run = get_active_run()
        a, b = (precheck.hold("gpt-4o", "hi", {"max_tokens": 100}) for _ in range(2))
        precheck.finish(b, failed=True)
        precheck.finish(a)
        checkpoint_sync()
        return run.reserved_usd
    assert agent() == 0


def check_refused_call_ends_run_breached():
    """An agent that catches the precheck's FencesStop and returns still ends breached, and says why."""
    from agentfences import precheck
    server = cloud(Server())
    try:
        @governed(budget_usd=0.001)
        def agent():
            try:
                precheck.hold("gpt-4o", "hi")
            except FencesStop as stop:
                return stop.message
        agent()
        assert server.of("end")[-1]["status"] == "breached"
        assert server.of("end")[-1]["breach"] == "budget_exceeded", server.of("end")[-1]
    finally:
        agentfences.init(local_only=True)


def check_guard_counts_model_calls_in_flight():
    """In cloud mode the server can't see this process's in-flight model calls, so the SDK checks them first."""
    from agentfences import precheck
    ran = []

    @tool(cost_usd=0.6)
    def pay():
        ran.append(1)

    server = cloud(Server(budget=100))  # the server alone would allow it
    try:
        @governed(budget_usd=1)
        def agent():
            run = get_active_run()
            precheck.hold("gpt-4o", "hi", {"max_tokens": 50_000})  # ~$0.50 in flight
            e = blocked(pay)
            return e, run.cost_usd + run.reserved_usd
        e, committed = agent()
        assert e and e.breach_type == "budget_exceeded" and not ran and committed <= 1, (e, committed)
        assert not [g for g in server.guards if g["phase"] == "check"], "refused before asking the server"
    finally:
        agentfences.init(local_only=True)


# ── Level 2: stops reach a stuck agent ────────────────────────────────────────

def check_stop_during_hanging_tool():
    server = cloud(Server(), stop_grace_s=0.2)
    try:
        @tool(name="crawl")
        async def hangs():
            await asyncio.sleep(60)

        @governed(budget_usd=1)
        async def agent():
            await checkpoint(cost_delta_usd=0.01)
            await hangs()

        def operator():
            wait_for(lambda: server.beats and server.beats[-1])
            server.heartbeat_commands = [{"run_id": server.beats[-1][0]["run_id"], "type": "stop"}]
        background(operator)
        started = time.monotonic()
        try:
            asyncio.run(agent())
        except FencesStop as e:
            assert e.breach_type == "stopped_by_user"
        else:
            raise AssertionError("a stopped run stuck in a tool should end with FencesStop")
        assert time.monotonic() - started < 3
        assert server.of("end")[-1]["status"] == "breached"
        assert any(r["stopped"] for beat in server.beats for r in beat), "the heartbeat says the stop was received"
    finally:
        agentfences.init(local_only=True)


def _stop_the_live_run(server):
    wait_for(lambda: server.beats and server.beats[-1])
    server.heartbeat_commands = [{"run_id": server.beats[-1][0]["run_id"], "type": "stop"}]


def check_stuck_sync_run_is_halted():
    """A sync run that never checkpoints after a stop is halted with FencesStop: at once in Python
    code (even under `except Exception`), or when a blocking call returns. A halt never lands later."""
    server = cloud(Server(), stop_grace_s=0.2)
    try:
        after = []

        @governed(budget_usd=1)
        def spins():
            checkpoint_sync(cost_delta_usd=0.01)
            while True:
                try:
                    sum(range(1000))
                except Exception:  # an agent's broad handler mustn't swallow the halt
                    pass

        @governed(budget_usd=1)
        def sleeps():
            checkpoint_sync(cost_delta_usd=0.01)
            while True:
                time.sleep(0.3)  # blocking: halted when it returns

        for agent in (spins, sleeps):
            background(lambda: _stop_the_live_run(server))
            started = time.monotonic()
            try:
                agent()
            except FencesStop as e:
                assert e.breach_type == "stopped_by_user"
            else:
                raise AssertionError(f"{agent.__name__}: a stuck sync run should be halted")
            after.append(round(time.monotonic() - started, 1))
            assert server.of("end")[-1]["status"] == "breached", server.of("end")[-1]
            server.beats.clear()
        assert all(t < 3 for t in after), after

        @governed(budget_usd=1)
        def inner():
            while True:
                sum(range(1000))

        @governed(budget_usd=1)
        def outer():
            checkpoint_sync(cost_delta_usd=0.01)
            inner()

        background(lambda: _stop_the_live_run(server))
        try:
            outer()
        except FencesStop:
            pass
        else:
            raise AssertionError("the outer run should be halted")
        inner_end, outer_end = server.of("end")[-2:]
        assert inner_end == {"status": "breached", "error": None, "exception": None, "breach": "stopped_by_user"}, inner_end
        assert outer_end["status"] == "breached", outer_end
        server.beats.clear()
        for _ in range(200_000):  # nothing is left pending in this thread
            pass

        # A halt sent just as the run's code finishes is withdrawn, never raised in the caller. The run's
        # thread is blocked in C, where a pending halt waits, while it's sent and then withdrawn.
        run = agentfences.core._start_run("racer", 1, None, None, None)
        gate, armed, outcome = threading.Lock(), threading.Event(), []
        gate.acquire()

        def run_code():
            run.thread = threading.get_ident()
            armed.set()
            try:
                gate.acquire()  # blocked here while the halt is sent and withdrawn
                for _ in range(200_000):
                    pass
                outcome.append("not halted")
            except agentfences.control.Halt:
                outcome.append("halted after disarm")
        worker = threading.Thread(target=run_code)
        worker.start()
        armed.wait()
        time.sleep(0.2)  # ponytail: let it reach gate.acquire(); a halt sent before then lands in Python code, as it should
        run.stopped, run.stopped_at = True, time.monotonic() - 1
        agentfences.control._halt(run)
        assert run.cancelled
        agentfences.control.disarm(run)
        gate.release()
        worker.join(5)
        assert outcome == ["not halted"], outcome
        agentfences.control.unregister(run)
    finally:
        agentfences.init(local_only=True)


def check_stuck_stream_is_cancelled():
    """A streaming run stuck in its own code after a stop is cancelled with FencesStop; a slow consumer isn't."""
    server = cloud(Server(), stop_grace_s=0.2)
    try:
        @governed(budget_usd=1)
        async def stream(stall_inside):
            await checkpoint(cost_delta_usd=0.01)
            yield 1
            if stall_inside:
                await asyncio.sleep(60)
            yield 2

        async def consume(stall_inside):
            got = []
            async for x in stream(stall_inside):
                got.append(x)
                if not stall_inside and x == 1:
                    await asyncio.sleep(1)  # the consumer is slow, not the stream: it's left alone
            return got

        background(lambda: _stop_the_live_run(server))
        started = time.monotonic()
        try:
            asyncio.run(consume(stall_inside=True))
        except FencesStop as e:
            assert e.breach_type == "stopped_by_user" and time.monotonic() - started < 3
        else:
            raise AssertionError("a stream stuck after a stop should end with FencesStop")
        assert server.of("end")[-1]["status"] == "breached"

        server.beats.clear()
        background(lambda: _stop_the_live_run(server))
        assert asyncio.run(consume(stall_inside=False)) == [1, 2]  # the stream itself never stalled
    finally:
        agentfences.init(local_only=True)


def check_stop_seen_at_checkpoint_is_not_cancelled():
    """An agent that sees its stop gets to wrap up: summarizing after the stop isn't cut off."""
    server = cloud(Server(), stop_grace_s=0.1)
    try:
        @governed(budget_usd=1)
        async def agent():
            for _ in range(500):
                result = await checkpoint(cost_delta_usd=0.001)
                if result.breached:
                    await asyncio.sleep(0.5)  # the closing summary
                    return result.breach_type
                await asyncio.sleep(0.01)

        def operator():
            wait_for(lambda: server.beats and server.beats[-1])
            server.heartbeat_commands = [{"run_id": server.beats[-1][0]["run_id"], "type": "stop"}]
        background(operator)
        assert asyncio.run(agent()) == "stopped_by_user"
    finally:
        agentfences.init(local_only=True)


def check_hanging_tool_times_out():
    @tool(timeout_s=0.1)
    async def stuck_async():
        await asyncio.sleep(60)

    @tool(timeout_s=0.1)
    def stuck_sync():
        time.sleep(2)

    for call in (lambda: asyncio.run(stuck_async()), stuck_sync):
        started = time.monotonic()
        try:
            call()
        except TimeoutError as e:
            assert "longer than 0.1s" in str(e)
        else:
            raise AssertionError("expected a timeout")
        assert time.monotonic() - started < 1


# ── Waiting at a limit (on_breach="hold") ─────────────────────────────────────

def held_beat(server, kind):
    """The run id of a heartbeat reporting a run held at a limit of this kind, or None."""
    for beat in reversed(server.beats):
        for r in beat:
            if r.get("held") == kind and r["paused"]:
                return r["run_id"]


def check_hold_at_a_breach_until_raised():
    server = cloud(Server())
    try:
        @governed(budget_usd=1, on_breach="hold")
        def agent():
            server.breach = "budget_exceeded"
            held = checkpoint_sync(cost_delta_usd=0.01)  # waits here until someone raises the limit
            after = checkpoint_sync(cost_delta_usd=0.01)
            return held, after

        def operator():
            wait_for(lambda: held_beat(server, "breach"))
            server.breach = None  # "Raise limits & resume": the run is back to running...
            server.heartbeat_commands = [{"run_id": held_beat(server, "breach"), "type": "resume", "id": "r1"}]  # ...and resumed
        background(operator)
        held, after = agent()
        assert held.ok and after.ok, (held, after)
        assert server.of("end")[-1]["status"] == "success"
    finally:
        agentfences.init(local_only=True)


def check_hold_ends_on_stop_or_timeout():
    server = cloud(Server(), pause_timeout_s=0.3)
    try:
        @governed(budget_usd=1, on_breach="hold")
        def timed_out():
            server.breach = "iteration_limit"
            r = checkpoint_sync(cost_delta_usd=0.01)
            server.breach = None  # raised later: the run is fine again, and mustn't be stuck as "paused"
            return r, checkpoint_sync(cost_delta_usd=0.01)
        r, later = timed_out()
        assert r.breach_type == "iteration_limit", "after the wait, the limit it was held at, not 'paused'"
        assert later.ok, later

        server = cloud(Server())

        @governed(budget_usd=1, on_breach="hold")
        def stopped():
            server.breach = "budget_exceeded"
            return checkpoint_sync(cost_delta_usd=0.01)

        def operator():
            wait_for(lambda: held_beat(server, "breach"))
            server.heartbeat_commands = [{"run_id": held_beat(server, "breach"), "type": "stop", "id": "s1"}]
        background(operator)
        assert stopped().breach_type == "stopped_by_user"
    finally:
        agentfences.init(local_only=True)


def check_hold_only_when_someone_can_raise_it():
    import time
    from agentfences import precheck

    @governed(budget_usd=0.01, on_breach="hold")
    def local():
        return checkpoint_sync(cost_delta_usd=0.02)
    assert local().breach_type == "budget_exceeded", "local mode never holds"

    @governed(budget_usd=0.001, on_breach="hold")
    def local_call():
        try:
            precheck.hold("gpt-4o", "hi", {"max_tokens": 1000})
        except FencesStop as stop:
            return stop.breach_type
    agentfences.init(local_only=True, pause_timeout_s=5)
    t = time.monotonic()
    assert local_call() == "budget_exceeded" and time.monotonic() - t < 1, "a refused call doesn't wait in local mode"
    agentfences.init(local_only=True)

    server = cloud(Server())
    server.reachable = False
    try:
        assert local().breach_type == "budget_exceeded", "a breach found while the server is unreachable never holds"
    finally:
        agentfences.init(local_only=True)
    try:
        agentfences.init(local_only=True, on_breach="wait")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown on_breach value should be refused")


def check_resume_releases_against_the_raised_limit():
    """A reply listing resume before limits (as the server does): the budget is raised before the resume lands."""
    from agentfences import control
    cloud(Server())
    try:
        @governed(budget_usd=0.01, on_breach="hold")
        def agent():
            run = get_active_run()
            seen = []
            real_set = run.resumed.set
            run.resumed.set = lambda: (seen.append(run.budget_usd), real_set())
            control.apply(run, [{"type": "resume", "id": "r1"}, {"type": "limits", "budget_usd": 5.0}])
            return seen
        assert agent() == [5.0], "the resume landed before the raised limit"
    finally:
        agentfences.init(local_only=True)


def check_hold_a_call_until_the_budget_is_raised():
    """A model call refused by the precheck, and a guarded tool denied for budget, wait for a raise too."""
    from agentfences import precheck
    ran = []

    @tool(cost_usd=0.5)
    def pay():
        ran.append(1)

    server = cloud(Server(budget=0.3))
    try:
        @governed(budget_usd=0.01, on_breach="hold")
        def agent():
            h = precheck.hold("gpt-4o", "hi", {"max_tokens": 1000})  # worst case just over the whole $0.01 budget: waits
            precheck.finish(h)
            pay()
            return h

        def operator():
            wait_for(lambda: held_beat(server, "budget"))
            rid = held_beat(server, "budget")
            server.heartbeat_commands = [{"run_id": rid, "type": "limits", "budget_usd": 5.0}, {"run_id": rid, "type": "resume", "id": "r1"}]
            server.beats.clear()
            wait_for(lambda: held_beat(server, "budget"))  # then the guarded tool: the server's budget is too low
            server.budget = 5.0
            server.heartbeat_commands = [{"run_id": rid, "type": "resume", "id": "r2"}]
        background(operator)
        assert agent() is not None and ran == [1]
        checks = [g for g in server.guards if g["phase"] == "check"]
        assert len({g["guard_id"] for g in checks}) == 1 and len(checks) >= 2, "re-checked with the same guard id"

        server = cloud(Server())

        @governed(budget_usd=0.01, on_breach="hold")
        async def async_agent():  # what instrument=True and the OpenAI Agents hooks use: waiting doesn't block the loop
            ticks = []
            ticker = asyncio.ensure_future(_tick(ticks))
            h = await precheck.hold_async("gpt-4o", "hi", {"max_tokens": 1000})
            ticker.cancel()
            return h, len(ticks)

        def release():
            wait_for(lambda: held_beat(server, "budget"))
            rid = held_beat(server, "budget")
            server.heartbeat_commands = [{"run_id": rid, "type": "limits", "budget_usd": 5.0}, {"run_id": rid, "type": "resume", "id": "r3"}]
        background(release)
        h, ticks = asyncio.run(async_agent())
        assert h is not None and ticks > 3, "the event loop kept running while the call waited"
    finally:
        agentfences.init(local_only=True)


async def _tick(ticks):
    while True:
        ticks.append(1)
        await asyncio.sleep(0.01)


# ── Level 1: limits hold when the server doesn't answer ───────────────────────

def check_network_partition():
    server = cloud(Server())
    server.reachable = False
    try:
        @governed(budget_usd=0.05)
        def agent():
            for step in range(100):
                if checkpoint_sync(cost_delta_usd=0.02).breached:
                    return step
        assert agent() == 2, "the budget is still enforced locally"

        cloud(Server(), fail_closed=True).reachable = False

        @governed(budget_usd=99)
        def strict():
            return checkpoint_sync(cost_delta_usd=0.01).breach_type
        assert strict() == "fences_unreachable"
    finally:
        agentfences.init(local_only=True)


if __name__ == "__main__":
    agentfences.init(local_only=True)
    checks = [
        check_parallel_calls_cannot_overspend, check_blocked_tools_never_run, check_failed_and_timed_out_calls_are_settled,
        check_guard_fails_closed, check_server_required_approval, check_mcp_session_calls_are_recorded_and_guarded,
        check_model_call_worst_case, check_parallel_model_calls_cannot_overspend, check_holds_are_released_exactly,
        check_refused_call_ends_run_breached, check_guard_counts_model_calls_in_flight,
        check_hold_at_a_breach_until_raised, check_hold_ends_on_stop_or_timeout, check_hold_only_when_someone_can_raise_it,
        check_hold_a_call_until_the_budget_is_raised, check_resume_releases_against_the_raised_limit,
        check_stop_during_hanging_tool, check_stuck_sync_run_is_halted, check_stuck_stream_is_cancelled,
        check_stop_seen_at_checkpoint_is_not_cancelled, check_hanging_tool_times_out,
        check_network_partition,
    ]
    for check in checks:
        check()
        print("ok", check.__name__)
    print("all enforcement checks passed")
