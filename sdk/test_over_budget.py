import asyncio
import fences
from fences import governed, checkpoint, BudgetExceeded

fences.init(api_key="fc_s2Swpy73aNBdpZtQinO6gY6BaEugHFEBdVO2y9y44VY", endpoint="https://fences-api-production.up.railway.app")


@governed(budget_usd=0.10)
async def expensive_agent():
    for i in range(20):
        await checkpoint(cost_delta_usd=0.02)  # will cross 0.10 on the 5th call
        print(f"  step {i} completed")
    return "should never reach here"


async def main():
    try:
        result = await expensive_agent()
        print("RESULT (unexpected):", result)
    except BudgetExceeded as e:
        print("CORRECTLY STOPPED:", e)


if __name__ == "__main__":
    asyncio.run(main())