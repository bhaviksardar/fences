# Fences

Runtime governance for AI agents. Give any agent budget limits, loop protection, and a decision audit trail in three lines of code.

```python
import fences
fences.init(local_only=True)

@fences.governed(budget_usd=0.50, max_iterations=20, max_duration_ms=60_000)
async def run_agent(query: str):
    response = call_llm(query)
    fences.log_decision(reasoning="searching for sources", action="web_search")
    await fences.checkpoint(cost_delta_usd=compute_cost(response))
    return response
```

If the agent exceeds its budget, loops past the iteration limit, or runs too long, `checkpoint()` raises and execution stops immediately.

## Install

```bash
pip install fences
```

## Quickstart — no backend needed

```python
import fences
from fences import governed, checkpoint, log_decision
from fences import BudgetExceeded, IterationLimitReached

fences.init(local_only=True)

@governed(budget_usd=0.10, max_iterations=10)
async def my_agent(query: str):
    for step in range(100):
        log_decision(reasoning=f"step {step}: searching", action="search")
        await checkpoint(cost_delta_usd=0.01)
    return "done"
```

## What gets enforced

| Limit | Parameter | Raises |
|---|---|---|
| Spend | `budget_usd` | `BudgetExceeded` |
| Iterations | `max_iterations` | `IterationLimitReached` |
| Duration | `max_duration_ms` | `TimeLimitReached` |

## Handling breaches

```python
from fences import BudgetExceeded, IterationLimitReached, TimeLimitReached

try:
    result = await my_agent("research this topic")
except BudgetExceeded as e:
    print(f"Stopped: spent ${e.spent_usd:.4f} of ${e.budget_usd:.4f}")
except IterationLimitReached as e:
    print(f"Stopped: {e.iterations} iterations reached")
except TimeLimitReached as e:
    print(f"Stopped: ran for {e.duration_ms}ms")
```

## Cloud mode

Connect to a Fences backend for persistent audit trails, a dashboard, and server-authoritative enforcement across distributed agents.

```python
fences.init(api_key="fc_...", endpoint="https://your-fences-instance.com")
```

Everything else stays the same — same decorator, same `checkpoint()` calls.

## License

MIT