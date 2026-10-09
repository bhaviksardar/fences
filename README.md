# (|=|) Fences

[![test](https://github.com/bhaviksardar/fences/actions/workflows/test.yml/badge.svg)](https://github.com/bhaviksardar/fences/actions/workflows/test.yml)

The open-source SDK for Fences, on-call for AI agents. It records what your agent decides and why, enforces limits the agent can't raise, and lets you stop a run from the Fences dashboard. Works offline too: local limits, no account.

```python
import agentfences
from agentfences import governed, checkpoint, log_decision

agentfences.init(local_only=True)

@governed(budget_usd=0.50, max_iterations=20, max_tokens=50_000)
async def run_agent(query: str, messages: list):
    while True:
        log_decision(reasoning="calling LLM", action="llm_call")
        response = call_llm(messages)

        result = await checkpoint(response)  # Fences reads the usage and prices the call

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
result = await checkpoint(response)

result.ok           # True if within all limits
result.breached     # True if a limit was crossed
result.breach_type  # "budget_exceeded" | "iteration_limit" | "time_limit" | "token_limit"
result.message      # "I've reached my budget limit... I'll summarise what I found."
result.system_prompt  # ready to inject into LLM conversation context
result.spent_usd    # the run's total spend so far
```

## Cost tracking

Pass the model's response to `checkpoint()` and Fences works out what the call cost. It reads the model name and token usage, including cache reads and writes and reasoning tokens, and prices them from a built-in table of current list prices:

```python
response = client.chat.completions.create(model="gpt-4o", messages=messages)
result = await checkpoint(response)
print(result.spent_usd)  # the run's total so far
```

- **Supported responses:** OpenAI (Chat Completions and Responses), Anthropic, Gemini and LangChain messages, or any object or dict with the same usage fields. If a gateway such as OpenRouter reports the billed cost, that is used instead.
- **Built-in prices:** about 290 models from OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI and Groq, including long-context pricing. They come from [LiteLLM's price list](https://github.com/BerriAI/litellm) and are refreshed each release.
- **Other models:** a model missing from the table still has its tokens counted, and Fences logs a warning once, because `budget_usd` can't stop a model it can't price. Add prices in USD per 1M tokens:
  ```python
  agentfences.init(local_only=True, prices={"my-finetune": {"input": 2.0, "output": 6.0}})
  ```
- **Extra costs:** add a paid tool call or anything else to the step with `checkpoint(response, cost_delta_usd=0.01)`. Without a response, `cost_delta_usd` is the step's whole cost.
- **Streaming:** pass the final message, or the last chunk once it carries usage (OpenAI: `stream_options={"include_usage": True}`).

## Quickstart — no backend needed

```python
import asyncio
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

print(asyncio.run(my_agent("hello")))
# I've reached my budget limit (spent $0.1200 of $0.1000). I'll summarize what I found so far.
```

## Limits

| Limit | Parameter | `result.breach_type` |
|---|---|---|
| Spend | `budget_usd` | `budget_exceeded` |
| Iterations | `max_iterations` | `iteration_limit` |
| Duration | `max_duration_ms` | `time_limit` |
| Tokens | `max_tokens` | `token_limit` |

A limit trips once it is exceeded, not when it is reached: a $0.10 budget allows exactly $0.10 of spend.

On Fences Cloud a run can also stop with `stopped_by_user` (someone pressed Stop in the dashboard), `key_daily_budget` or `key_monthly_budget` (the agent's daily or monthly cap across all its runs). With `init(fail_closed=True)`, an unreachable backend stops it with `fences_unreachable`. Each comes with its own `message` and `system_prompt`.

## Sync agents and streaming

`@governed` works on sync functions, async functions and async generators. Sync agents use `checkpoint_sync()`, which takes the same arguments and returns the same `CheckpointResult`:

```python
from agentfences import governed, checkpoint_sync

@governed(budget_usd=0.50)
def my_agent(query: str):
    while True:
        response = call_llm(query)
        result = checkpoint_sync(response)
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

In cloud mode the backend is authoritative: it sees spend from every process sharing a run, so one process can't overspend because it didn't see another's spend. On Fences Cloud, a limit raised from the dashboard lets the run's next `checkpoint()` pass, so a fenced agent can resume. Backend calls run off the event loop and `log_decision()` never blocks; decisions are sent in the background and flushed at exit (call `agentfences.flush()` yourself before a serverless handler returns).

If the backend can't be reached, limits are enforced locally and a warning is logged once per run. To treat an unreachable backend as a breach instead:
```python
agentfences.init(api_key="fc_...", endpoint="https://...", fail_closed=True)
```

## Project layout

```
fences/
├── docs/       landing page
└── sdk/        agentfences Python package
    └── agentfences/
```

## License

MIT
