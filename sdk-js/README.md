# agentfences (TypeScript / JavaScript)

The TypeScript SDK for [Fences](https://github.com/bhaviksardar/fences), on-call for AI agents: limits your agents can't raise, cost tracking from real usage, a decision trail, and a stop button. Works offline too: local limits, no account. Node 18+, no dependencies.

```bash
npm install agentfences
```

## Quickstart (no account needed)

```ts
import { init, governed, checkpoint, logDecision } from "agentfences";

init({ localOnly: true });

const research = governed(async (question: string) => {
  for (;;) {
    logDecision("asking the model for the next step", "llm_call");
    const response = await openai.chat.completions.create({ model: "gpt-4o", messages });
    const result = await checkpoint(response);   // Fences reads the usage and prices the call
    if (result.breached) return result.message;  // "I've reached my budget limit (spent $0.5100 of $0.5000)..."
    // ...
  }
}, { budgetUsd: 0.5, maxIterations: 20 });
```

`checkpoint()` never throws for a breach: it returns a result with `ok`, `breached`, `breachType`, `message` (ready for the user), `systemPrompt` (to give the model so it wraps up) and `spentUsd`.

## Cost tracking

Pass the model's response to `checkpoint()` and Fences works out what the call cost from its model and token usage, including cache reads and writes. It understands results from the OpenAI and Anthropic SDKs, the Vercel AI SDK (`generateText`), Gemini and LangChain.js, or any object with the same usage fields. Prices come from a built-in table of about 290 models (the same one as the Python SDK).

- Unknown model: its tokens are counted and Fences warns once. Add prices in USD per 1M tokens: `init({ localOnly: true, prices: { "my-finetune": { input: 2, output: 6 } } })`. Both prices are required; an invalid one throws at `init()`.
- Extra costs, or no response at all: `checkpoint(response, { costDeltaUsd: 0.01 })` or `checkpoint(null, { costDeltaUsd: 0.02, tokensUsed: 450 })`.

## Limits

`budgetUsd`, `maxIterations` (default 100), `maxDurationMs` (default 5 minutes) and `maxTokens` (default none). All are optional. A limit trips once it's exceeded, not when reached. With no budget anywhere, the run's spend isn't capped and Fences warns once per agent. On Fences Cloud, limits left out come from the agent's page in the dashboard.

## Tools, context, errors and redaction

```ts
import { tool, context } from "agentfences";

const webSearch = tool(async ({ query }: { query: string }) => search(query));   // each call recorded as a span

await context({ user_id: "u_123", session_id: "s_9" }, () => research(question));  // tags every run inside
```

- Tool and model calls are recorded as OpenTelemetry-shaped spans (`execute_tool webSearch`, `gen_ai.*` attributes), on `getActiveRun().events`. A tool call carries its argument names (`query, limit`), not their values; `init({ toolArguments: "full" })` records the values as JSON (300 characters at most), `"none"` neither (from 0.9.0). A guarded tool's check and an approval request still send the values.
- When a governed run throws, Fences records the error's type, message and stack with the run; the error still reaches your code.
- `init({ redact: (event) => ... })` sees everything before it leaves the process and can change or drop it.

## Fences Cloud

```ts
init({ apiKey: process.env.FENCES_API_KEY, endpoint: "https://your-fences-host", environment: "prod", release: "v1.4.2" });
```

The server is authoritative for limits and sees spend across processes. If it can't be reached, limits are enforced locally; pass `failClosed: true` to stop instead. `endpoint` must be an `https://` URL unless the server is on the same machine; `allowHttp: true` permits plain HTTP for a trusted private network (from 0.9.1). A server reply can raise a budget, but can't make a budget set in code unlimited. A rejected key throws `FencesAuthError` when the run starts, and a key revoked during a run ends it at its next checkpoint (`fences_unreachable`); a quarantined agent throws `AgentQuarantined` before any of its code runs. A cost that isn't a finite number of zero or more throws instead of being counted. In serverless handlers, `await flush()` before returning so queued decisions and spans are sent.

## Live control (from 0.4.0)

In cloud mode a heartbeat (every 15 seconds, `init({ heartbeatS })`) lets whoever is on call act on a running agent from the dashboard:

- **Stop:** the next `checkpoint()` returns `breachType: "stopped_by_user"`, so the agent wraps up gracefully. If the run doesn't reach a checkpoint within `stopGraceS` (default 30) seconds, say it's stuck in a tool, the governed call rejects with `FencesStop` and the run's `signal` aborts (from 0.5.0). JavaScript can't stop running code, so pass `getActiveRun()?.signal` to `fetch` and other cancellable work for a stuck run to really stop. `init({ stopGraceS: null })` turns this off.
- **Pause and resume:** a paused run waits inside its next `checkpoint()` until it's resumed; after `pauseTimeoutS` (default one hour) it returns `breachType: "paused"`.
- **Limits** changed in the dashboard apply to the run immediately. **Quarantine** makes the governed call throw `AgentQuarantined` before any agent code runs; runs already going hold at their next checkpoint, and their guarded tools are blocked.

An agent can ask a person before doing something risky:

```ts
import { requestApproval } from "agentfences";

const approval = await requestApproval("Refund $240 to order 1182?", { amountUsd: 240, timeoutS: 600 });
if (!approval.granted) return `I didn't issue the refund: ${approval.note}`;
```

An approved amount raises the run's budget. It never hangs or throws: in local mode, outside a run, or on a server without approvals it's denied right away, and with no answer it's denied after `timeoutS`. The heartbeat's timer never keeps your process alive on its own.

**Waiting at a limit** (from 0.6.0): with `init({ onBreach: "hold" })` (or `governed(fn, { onBreach: "hold" })`), a limit the server says is crossed makes `checkpoint()` wait instead of returning the breach, so whoever is on call can raise it from the dashboard and the agent carries on. A guarded tool denied for budget waits the same way. A stop ends the wait; after `pauseTimeoutS` the agent gets the breach it was held at. Cloud only. Don't hold an agent a person is waiting on.

**Containment hooks** (from 0.5.0) run in your process when a stop or quarantine arrives, to contain what Fences can't reach, such as a credential the agent holds:

```ts
init({ apiKey: "fc_...", onQuarantine: async (event) => vault.revoke(`agent/${event.agentName}`), onStop: (event) => { ... } });
```

`onStop` runs once per run, and `onQuarantine` once per quarantine however many runs it reaches. Each gets `{ type, runId, agentName, id, by, reason }`. Fences records whether it worked, threw (and why) or ran past 30 seconds. A process that's about to exit waits for its hooks first; in serverless handlers, `await flush()` does the same. The SDK also acknowledges every dashboard command it receives, so the dashboard can tell "the agent has it" from "the agent hasn't acknowledged".

## Guarded tools (from 0.5.0)

For actions with consequences (refunds, payments, writes to production), guard the tool, and it asks Fences **before** it runs:

```ts
import { tool, ToolBlocked } from "agentfences";

const issueRefund = tool(refund, { costUsd: (orderId: string, amount: number) => amount, approve: true });
const deploy = tool(deployImpl, { guard: true, timeoutS: 30 });
```

- The tool runs only if its run isn't stopped, paused, over a limit or quarantined. Otherwise it throws `ToolBlocked` (with `reason` and `breachType`) and its code never runs.
- `costUsd` (a number, or a function of the tool's arguments) is **reserved** before the call and charged after, so parallel calls can't overspend the budget together. A call that throws is released uncharged; one that times out is charged, because it may have happened.
- `approve: true` asks a person on the dashboard before each call. The dashboard can also require approval for a tool, and the SDK then asks for it.
- **Guards fail closed:** if Fences can't be reached, or doesn't support guarded tools, the tool is blocked. In local mode the same rules apply in-process, and `approve` is always denied.
- `timeoutS` works on any tool: past it, the call rejects with a `TimeoutError`. The tool's own code carries on unless it uses `getActiveRun()?.signal`.

A guarded or timed tool always returns a promise. What each part guarantees: [ENFORCEMENT.md](../ENFORCEMENT.md).

**MCP tools** (from 0.7.0): wrap an MCP `Client` and every `callTool` is recorded, with no `tool()` per tool. The tools you name are guarded as above:

```ts
import { mcp } from "agentfences";

mcp.wrap(client, { guard: ["send_email"], costUsd: { refund: (args) => args.amount as number } });
```

`guard: true` guards every tool, and `approve: [...]` names tools a person approves first. A result with `isError` is recorded as a failed call and still comes back to the agent.

## Model calls (from 0.8.0)

`instrument(client)` records every OpenAI and Anthropic call made inside a governed run (model, tokens, cost, latency, errors) and checks it against the budget **before** it's sent:

```ts
import OpenAI from "openai";
import { instrument } from "agentfences";

const openai = instrument(new OpenAI());  // one call covers every client of that SDK
```

- If a call's worst case could take the run past its budget, it rejects with `FencesStop` (`budget_exceeded`) and never reaches the provider. The worst case is the request's input, estimated high, plus its output cap: `max_tokens` (or `max_completion_tokens`, `max_output_tokens`) if set, otherwise the model's whole output limit, so **set `max_tokens` on calls in runs with small budgets**.
- Each call's worst case is held while it's in flight, so parallel calls can't overspend together. A failed call is released; a finished one is replaced by its actual cost, and the next `checkpoint()` releases it. With `onBreach: "hold"`, a call that doesn't fit waits for someone to raise the budget.
- Calls are recorded, not counted as steps: keep calling `checkpoint(response)`, which adds the cost and checks the other limits.
- The SDK's own promise comes back, with `.withResponse()` and `.asResponse()`. After `.asResponse()` the call is recorded without usage, and its worst case stays held until the next checkpoint.
- OpenAI: `chat.completions.create` and `responses.create`, including OpenAI-compatible clients. Anthropic: `messages.create`. Streams are recorded when they start; pass the final usage to `checkpoint()`.

## Not yet in the TypeScript SDK

The Python SDK also integrates with LangGraph and the OpenAI Agents SDK. Model calls made without `instrument(client)` are checked afterwards, by `checkpoint()`, so a run can go over its budget by one such call. Everything here follows the same server protocol ([PROTOCOL.md](../PROTOCOL.md)).
