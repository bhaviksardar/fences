"""
Offline checks for agentfences in local mode: no server, no API key, no network.

    pip install ./sdk && python tests/test_local.py
"""
import asyncio
import socket

import agentfences
from agentfences import (
    governed, checkpoint, checkpoint_sync, log_decision, get_active_run,
    BudgetExceeded, IterationLimitReached, TimeLimitReached, TokenLimitReached,
)


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


if __name__ == "__main__":
    check_requires_init()
    agentfences.init(local_only=True)
    checks = [
        check_readme_quickstart, check_limits_trip_once_exceeded, check_time_limit,
        check_raise_on_breach, check_sync_agent, check_streaming_agent,
        check_concurrent_runs_are_independent, check_nested_runs, check_outside_a_run,
    ]
    for check in checks:
        check()
        print("ok", check.__name__)
    agentfences.flush()
    print("all local checks passed")
