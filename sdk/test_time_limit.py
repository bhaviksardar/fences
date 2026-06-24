import asyncio
import fences
from fences import governed, checkpoint, TimeLimitReached

fences.init(api_key="fc_hIOpZmo0j7hnASlUCx7p4YAgIYhGUoXcCNYIW_GRfCo", endpoint="http://localhost:8000")


@governed(budget_usd=99.0, max_iterations=1000, max_duration_ms=500)  # 500ms limit
async def slow_agent():
    for i in range(100):
        await asyncio.sleep(0.1)  # each step takes 100ms — hits 500ms limit around step 5
        await checkpoint(cost_delta_usd=0.00)
        print(f"  step {i} completed")
    return "should never reach here"


async def main():
    try:
        await slow_agent()
        print("FAILED: agent ran to completion, should have been stopped")
    except TimeLimitReached as e:
        print(f"CORRECTLY STOPPED: {e}")


if __name__ == "__main__":
    asyncio.run(main())