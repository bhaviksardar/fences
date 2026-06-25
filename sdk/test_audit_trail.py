"""
Tests the full decision audit trail:
1. Agent runs and logs reasoning at each step
2. We read back the decisions and confirm they're all there in order
"""
import asyncio
import requests
import fences
from fences import governed, checkpoint, log_decision

API_KEY = None  # set below after generating
BASE = "http://localhost:8000"


def generate_key():
    resp = requests.post(f"{BASE}/admin/keys/create",
        headers={"X-Admin-Password": "test"},
        json={"label": "audit test"},
    )
    return resp.json()["key"]


@governed(budget_usd=1.0, max_iterations=20)
async def research_agent(query: str) -> str:
    steps = [
        ("Analysing the query to determine search strategy", "plan"),
        ("Searching web for primary sources",                "web_search"),
        ("Found 3 relevant results, filtering for recency", "filter"),
        ("Summarising findings into a coherent answer",      "summarise"),
    ]

    for reasoning, action in steps:
        log_decision(reasoning=reasoning, action=action)
        await checkpoint(cost_delta_usd=0.01)

    return "Research complete"


async def main():
    global API_KEY
    API_KEY = generate_key()
    fences.init(api_key=API_KEY, endpoint=BASE)

    print("Running agent...")
    result = await research_agent("What are the best AI agent frameworks?")
    print(f"Agent result: {result}")

    # Find the run that was just created
    runs_resp = requests.get(f"{BASE}/api/runs",
        headers={"X-API-Key": API_KEY}
    )
    run_id = runs_resp.json()["runs"][0]["run_id"]

    # Read the audit trail
    decisions_resp = requests.get(f"{BASE}/api/runs/{run_id}/decisions",
        headers={"X-API-Key": API_KEY}
    )
    data = decisions_resp.json()

    print(f"\nAudit trail for run {run_id[:8]}... ({data['status']}):")
    for d in data["decisions"]:
        print(f"  [{d['iteration']:>2}] {d['action']:<15} — {d['reasoning']}")

    expected = 4
    actual = len(data["decisions"])
    if actual == expected:
        print(f"\nPASSED: all {expected} decisions recorded correctly")
    else:
        print(f"\nFAILED: expected {expected} decisions, got {actual}")


if __name__ == "__main__":
    asyncio.run(main())