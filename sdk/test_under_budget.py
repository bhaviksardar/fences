import asyncio
import fences
from fences import governed, checkpoint, BudgetExceeded

fences.init(api_key="fc_s2Swpy73aNBdpZtQinO6gY6BaEugHFEBdVO2y9y44VY", endpoint="https://fences-api-production.up.railway.app")


@governed(budget_usd=0.10)
async def cheap_agent():
    for i in range(3):
        await checkpoint(cost_delta_usd=0.02)  # total: 0.06, under 0.10
    return "done, stayed under budget"


async def main():
    result = await cheap_agent()
    print("RESULT:", result)


if __name__ == "__main__":
    asyncio.run(main())