# Fences

Runtime governance for AI agents. Budget limits, loop protection, token governance, and a decision audit trail — in three lines of code.

```python
import agentfences
from agentfences import governed, checkpoint, log_decision

agentfences.init(local_only=True)

@governed(budget_usd=0.50, max_iterations=20, max_tokens=50_000)
async def run_agent(query: str, messages: list):
    while True:
        log_decision(reasoning="calling LLM", action="llm_call")
        response = call_llm(messages)

        result = await checkpoint(
            cost_delta_usd=compute_cost(response.usage),
            tokens_used=response.usage.total_tokens,
        )

        if result.breached:
            # Agent handles the limit gracefully — no exception, no crash
            messages.append({"role": "system", "content": result.system_prompt})
            return call_llm(messages)  # LLM summarises what it found

        if response.is_done():
            return response
```

## Install

```bash
pip install agentfences
```

## How it works

`checkpoint()` returns a `CheckpointResult` after every call — not an exception. The agent reads it and decides what to do: stop, summarise, ask for more budget. No try/catch in your orchestration layer.

```python
result = await checkpoint(cost_delta_usd=0.02, tokens_used=450)

result.ok           # True if within all limits
result.breached     # True if a limit was crossed
result.breach_type  # "budget_exceeded" | "iteration_limit" | "time_limit" | "token_limit"
result.message      # "I've reached my budget limit... I'll summarise what I found."
result.system_prompt  # ready to inject into LLM conversation context
```

## Quickstart — no backend needed

```python
import agentfences
from agentfences import governed, checkpoint, log_decision

agentfences.init(local_only=True)  # no account, no API key

@governed(budget_usd=0.10, max_iterations=5, max_tokens=2000)
async def my_agent(query: str):
    for step in range(100):
        log_decision(reasoning=f"step {step}: searching", action="search")

        result = await checkpoint(cost_delta_usd=0.02, tokens_used=200)
        if result.breached:
            return result.message  # or inject result.system_prompt into your LLM

    return "done"
```

## Limits

| Limit | Parameter | `result.breach_type` |
|---|---|---|
| Spend | `budget_usd` | `budget_exceeded` |
| Iterations | `max_iterations` | `iteration_limit` |
| Duration | `max_duration_ms` | `time_limit` |
| Tokens | `max_tokens` | `token_limit` |

A limit trips once it is exceeded, not when it is reached: a $0.10 budget allows exactly $0.10 of spend.

## Legacy exception mode

If you prefer exceptions over result objects:

```python
@governed(budget_usd=0.50, raise_on_breach=True)
async def my_agent():
    ...
    await checkpoint(cost_delta_usd=0.02)  # raises BudgetExceeded, IterationLimitReached, etc.
```

## Sync agents and streaming

`@governed` works on sync functions, async functions and async generators. Sync agents use `checkpoint_sync()`, which takes the same arguments and returns the same `CheckpointResult`:

```python
from agentfences import governed, checkpoint_sync

@governed(budget_usd=0.50)
def my_agent(query: str):
    while True:
        response = call_llm(query)
        result = checkpoint_sync(cost_delta_usd=0.02, tokens_used=response.usage.total_tokens)
        if result.breached:
            return result.message
```

A governed async generator (a streaming agent) stays governed until the stream is exhausted. Concurrent agents in one event loop and nested `@governed` calls each get their own run.

## Modes

**Local (free)** — governance runs entirely in-process. No account, no backend, no API key.
```python
agentfences.init(local_only=True)
```

**Cloud (coming soon)** — persistent audit trails, live dashboard, server-authoritative enforcement. Same code, one line changes.
```python
agentfences.init(api_key="fc_...", endpoint="https://your-fences-instance.com")
```

In cloud mode the backend is authoritative: it sees spend from every process sharing a run, and a limit raised from the dashboard lets the run's next `checkpoint()` pass, so a fenced agent can resume. Backend calls run off the event loop and `log_decision()` never blocks; decisions are sent in the background and flushed at exit (call `agentfences.flush()` yourself before a serverless handler returns).

If the backend can't be reached, limits are enforced locally and a warning is logged once per run. To treat an unreachable backend as a breach instead:
```python
agentfences.init(api_key="fc_...", endpoint="https://...", fail_closed=True)
```

## License

MIT