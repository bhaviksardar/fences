"""
Fires 20 concurrent checkpoint calls of $0.01 each at the same run.
If the atomic DB update is working, the final total should be exactly $0.20.
If there were a race condition (read-modify-write done in Python instead of
in the database), some increments would get silently lost and the total
would come out lower than expected.
"""
import asyncio
import aiohttp

API_KEY = "ag-dev-key"
BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}
RUN_ID = "concurrency-test-001"
N_CHECKPOINTS = 20
COST_PER_CHECKPOINT = 0.01


async def start_run(session):
    async with session.post(f"{BASE}/api/runs/start", json={
        "run_id": RUN_ID, "agent_name": "concurrent_agent", "budget_usd": 999.0  # high enough to not breach mid-test
    }, headers=HEADERS) as resp:
        return await resp.json()


async def checkpoint(session, i):
    async with session.post(f"{BASE}/api/runs/{RUN_ID}/checkpoint", json={
        "cost_delta_usd": COST_PER_CHECKPOINT
    }, headers=HEADERS) as resp:
        return await resp.json()


async def get_run(session):
    async with session.get(f"{BASE}/api/runs/{RUN_ID}", headers=HEADERS) as resp:
        return await resp.json()


async def main():
    async with aiohttp.ClientSession() as session:
        await start_run(session)

        # Fire all checkpoints at once, truly concurrently
        results = await asyncio.gather(*[checkpoint(session, i) for i in range(N_CHECKPOINTS)])

        final = await get_run(session)
        expected = round(N_CHECKPOINTS * COST_PER_CHECKPOINT, 4)
        actual = round(final["run"]["spent_usd"], 4)

        print(f"Expected total: ${expected}")
        print(f"Actual total:   ${actual}")

        if actual == expected:
            print(f"PASSED: all {N_CHECKPOINTS} concurrent increments were counted correctly")
        else:
            lost = round((expected - actual) / COST_PER_CHECKPOINT)
            print(f"FAILED: {lost} increment(s) were lost to a race condition")


if __name__ == "__main__":
    asyncio.run(main())