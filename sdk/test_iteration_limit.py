import asyncio
import fences
from fences import governed, checkpoint, IterationLimitReached

fences.init(api_key="fc_s2Swpy73aNBdpZtQinO6gY6BaEugHFEBdVO2y9y44VY", endpoint="http://localhost:8000")


@governed(budget_usd=99.0, max_iterations=5)
async def runaway_agent():
    for i in range(100):  # tries to loop 100 times, limit is 5
        await checkpoint(cost_delta_usd=0.00)
        print(f"  iteration {i} completed")
    return "should never reach here"


async def main():
    try:
        await runaway_agent()
        print("FAILED: agent ran to completion, should have been stopped")
    except IterationLimitReached as e:
        print(f"CORRECTLY STOPPED: {e}")


if __name__ == "__main__":
    asyncio.run(main())