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

Every limit is optional. On Fences Cloud, a limit left out of `@governed(...)` comes from the agent's page in the dashboard, so `@governed()` on its own works. Limits set nowhere default to 100 steps, 5 minutes and no token limit, and no budget: the run still starts, but its spend isn't capped, and Fences logs a warning once per agent (and flags the agent in the dashboard).

On Fences Cloud a run can also stop with `stopped_by_user` (someone pressed Stop in the dashboard), `paused` (paused and not resumed in time), `key_daily_budget` or `key_monthly_budget` (the agent's daily or monthly cap across all its runs). With `init(fail_closed=True)`, an unreachable backend stops it with `fences_unreachable`. Each comes with its own `message` and `system_prompt`.

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

## Frameworks: LangChain, LangGraph and the OpenAI Agents SDK

Agents built on a framework need no decorators. Each model call is recorded with its tokens and cost and checkpointed, so limits apply, and each tool call is recorded. Limits work as in `@governed` and are all optional.

Callbacks and hooks can't hand a result back to the agent, so when a limit is crossed these integrations raise `FencesStop`. It carries the same `breach_type`, `message` and `system_prompt` as a breached `checkpoint()`.

**LangChain and LangGraph** (needs `langchain-core`):

```python
from agentfences import FencesStop
from agentfences.langchain import FencesCallbackHandler

try:
    graph.invoke(inputs, config={"callbacks": [FencesCallbackHandler(budget_usd=2)]})
except FencesStop as stop:
    answer = stop.message
```

The outermost run becomes a Fences run named after it (or `agent_name=...`), and joins the run if it's already inside `@governed`. Sync and async graphs both work. LangChain logs its own "Error in FencesCallbackHandler.on_llm_end callback" line when the handler stops a graph; that's expected.

**OpenAI Agents SDK** (needs `openai-agents`, Python 3.10+):

```python
from agentfences import FencesStop, openai_agents

try:
    result = await openai_agents.run(agent, "Research cats", budget_usd=2)   # Runner.run, governed
except FencesStop as stop:
    answer = stop.message
```

Already inside `@governed`? Use `Runner.run(agent, input, hooks=openai_agents.FencesRunHooks())`.

With a framework integration, don't also pass `instrument=True` or decorate tools with `@agentfences.tool`: each call would be recorded twice.

## Live control

In cloud mode the SDK keeps a heartbeat with Fences, so whoever is on call can act on a run while it's going:

- **Stop:** the run's next `checkpoint()` returns `breach_type="stopped_by_user"` with its message, so the agent wraps up gracefully. A stop sticks, whatever later responses say.
- **Pause and resume:** a paused run waits inside its next `checkpoint()` until it's resumed, then carries on. If no one resumes it within `pause_timeout_s` (default one hour), the checkpoint returns `breach_type="paused"`.
- **Change limits:** new limits apply to the run immediately.
- **Quarantine:** calling a governed function for a quarantined agent raises `AgentQuarantined` before any of its code runs.

An agent can also ask a person before doing something risky:

```python
approval = await agentfences.request_approval("Refund $240 to order 1182?", amount_usd=240, timeout_s=600)
if not approval.granted:
    return f"I didn't issue the refund: {approval.note}"
```

The answer has `granted`, `by`, `note` and `amount_usd`; an approved amount raises the run's budget by that much. `request_approval_sync()` does the same for synchronous agents. It never hangs and never raises: in local mode, outside a run, or when the server can't take requests it returns denied right away, and with no answer it returns denied after `timeout_s`.

The heartbeat reports live runs every 15 seconds (`init(heartbeat_s=...)`), and every 2 seconds while a run is paused or waiting for approval. A Fences server without these features gets one warning: stops still apply at the next checkpoint, and approval requests are denied.

## Evidence: tool calls and model calls

On-call can only explain what was recorded. Fences records two kinds of call on the run automatically, alongside your `log_decision()` entries.

**Tool calls.** Decorate your agent's tools:

```python
@agentfences.tool
def web_search(query: str, limit: int = 5) -> list: ...

@agentfences.tool(name="fetch")
async def fetch_page(url: str) -> str: ...
```

Each call records the tool's name, a short summary of its arguments (`query='cats', limit=5`), whether it succeeded (and the error if not) and how long it took. Exceptions still reach your code, and outside a governed run the tool is just called.

**Model calls.** Pass `instrument=True` to record every OpenAI and Anthropic call made inside a governed run, with no other code changes:

```python
agentfences.init(api_key="fc_...", endpoint="https://...", instrument=True)
```

Each call records the model, tokens (including cache reads and writes), cost, latency, and the error and status code if it failed (a 429 rate limit, say). This is a record only: it doesn't count as a checkpoint or add to the run's spend, so keep calling `checkpoint(response)` where limits should be checked. Streamed calls are recorded without token counts.

Both kinds of call are kept on the run (`get_active_run().events`, newest 1,000) and, in cloud mode, sent to Fences in the background in batches. They pass through the `redact` hook first as `tool_call` (`name`, `args`, `error`) and `llm_call` (`model`, `error`) events. A Fences server that doesn't accept them yet gets a single warning and nothing else changes.

## Context, errors and redaction

Tag runs so on-call can see which deploy and which customer a run belongs to:

```python
agentfences.init(api_key="fc_...", endpoint="https://...", environment="prod", release="v1.4.2")

with agentfences.context(user_id="u_123", session_id="s_9", trace_id=span_id):
    await support_agent(question)   # every run started in here carries these values
```

Blocks nest (inner values win), and `get_active_run().context` shows a run's values from inside the agent.

When a governed run raises, Fences records the exception type, message and the last 15 stack frames with the run, and its error reads `ValueError: page 7 returned 403` rather than just the message. The exception still reaches your code as usual.

Everything that leaves your process (a run's context, decisions, and a failed run's error) passes through an optional `redact` hook first. Return the event, changed or not, or `None` to drop it:

```python
def redact(event):
    if event["type"] == "decision":
        event["reasoning"] = EMAIL.sub("[email]", event["reasoning"])
    return event

agentfences.init(api_key="fc_...", endpoint="https://...", redact=redact)
```

Events have a `type` of `run_start` (`context`), `decision` (`reasoning`, `action`), `tool_call` (`name`, `args`, `error`), `llm_call` (`model`, `error`) or `run_end` (`status`, `error`, `exception`). If the hook raises, that event is dropped rather than sent unredacted, and a warning is logged once. Runs still start and end, so limits keep working.

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
