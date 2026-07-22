"""
Verifies decisions are preserved even when a run gets breached.
This is the important case — you need the audit trail most when
something went wrong, not when it completed successfully.
"""
import asyncio
import requests
import fences
from fences import governed, checkpoint, log_decision, IterationLimitReached

API_KEY = "fc_s2Swpy73aNBdpZtQinO6gY6BaEugHFEBdVO2y9y44VY"
BASE = "https://fences-api-production.up.railway.app"

fences.init(api_key=API_KEY, endpoint=BASE)


def generate_key():
    """Kept for local testing only — not used when API_KEY is set above."""
    resp = requests.post(f"{BASE}/admin/keys/create",
        headers={"X-Admin-Password": "test"},
        json={"label": "breach audit test"},
    )
    return resp.json()["key"]


@governed(budget_usd=1.0, max_iterations=3)
async def runaway_agent() -> str:
    for i in range(100):
        log_decision(
            reasoning=f"Step {i}: retrying because tool returned empty result",
            action="retry_tool_call"
        )
        await checkpoint(cost_delta_usd=0.00)
    return "never reaches here"


async def main():
    print("Running agent (expect it to be stopped at iteration 3)...")
    try:
        await runaway_agent()
    except IterationLimitReached as e:
        print(f"Agent stopped: {e}")

    # Fetch decisions from the breached run
    runs_resp = requests.get(f"{BASE}/api/runs", headers={"X-API-Key": API_KEY})
    run = runs_resp.json()["runs"][0]
    run_id = run["run_id"]

    decisions_resp = requests.get(
        f"{BASE}/api/runs/{run_id}/decisions",
        headers={"X-API-Key": API_KEY}
    )
    data = decisions_resp.json()

    print(f"\nAudit trail for breached run {run_id[:8]}... (status: {data['status']}):")
    for d in data["decisions"]:
        print(f"  [{d['iteration']:>2}] {d['action']:<20} — {d['reasoning']}")

    if data["status"] == "breached" and len(data["decisions"]) > 0:
        print(f"\nPASSED: {len(data['decisions'])} decisions preserved on breached run")
    else:
        print("\nFAILED: decisions not preserved on breach")


if __name__ == "__main__":
    asyncio.run(main())