"""
Simulates the scenario both local AND server enforcement was meant to catch:
two processes (or a restarted process) sharing the same run_id, where the
server knows about spend that this local process's RunState doesn't.

If server-side enforcement isn't actually authoritative, this agent would
keep running past budget because its LOCAL tally looks fine.
"""
import asyncio
import requests
import fences
from fences import governed, checkpoint, BudgetExceeded

fences.init(api_key="ag-dev-key", endpoint="http://localhost:8000")

RUN_ID = "shared-run-test-001"


def simulate_other_process_spend():
    """Pretend a different process already spent $0.09 of this run's budget."""
    requests.post(
        "http://localhost:8000/api/runs/start",
        json={"run_id": RUN_ID, "agent_name": "expensive_agent", "budget_usd": 0.10},
        headers={"X-API-Key": "ag-dev-key"},
    )
    requests.post(
        f"http://localhost:8000/api/runs/{RUN_ID}/checkpoint",
        json={"cost_delta_usd": 0.09},
        headers={"X-API-Key": "ag-dev-key"},
    )


async def main():
    simulate_other_process_spend()  # server now shows $0.09 / $0.10 spent

    # Manually construct local state that's UNAWARE of that $0.09 —
    # this mimics a process that didn't see the other spend.
    from fences.core import RunState, _set_active_run, _require_client
    run = RunState(run_id=RUN_ID, agent_name="expensive_agent", budget_usd=0.10)
    run.cost_usd = 0.0  # local thinks nothing has been spent yet
    _set_active_run(run)

    try:
        # Locally this looks totally fine (0.0 -> 0.02, way under 0.10).
        # But the SERVER knows the true total is 0.09 + 0.02 = 0.11 — over budget.
        await checkpoint(cost_delta_usd=0.02)
        print("FAILED TEST: local-only check let this through incorrectly")
    except BudgetExceeded as e:
        print("PASSED: server caught what local enforcement missed —", e)


if __name__ == "__main__":
    asyncio.run(main())