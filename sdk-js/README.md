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

- Unknown model: its tokens are counted and Fences warns once. Add prices in USD per 1M tokens: `init({ localOnly: true, prices: { "my-finetune": { input: 2, output: 6 } } })`.
- Extra costs, or no response at all: `checkpoint(response, { costDeltaUsd: 0.01 })` or `checkpoint(null, { costDeltaUsd: 0.02, tokensUsed: 450 })`.

## Limits

`budgetUsd`, `maxIterations` (default 100), `maxDurationMs` (default 5 minutes) and `maxTokens` (default none). All are optional. A limit trips once it's exceeded, not when reached. With no budget anywhere, the run's spend isn't capped and Fences warns once per agent. On Fences Cloud, limits left out come from the agent's page in the dashboard.

## Tools, context, errors and redaction

```ts
import { tool, context } from "agentfences";

const webSearch = tool(async ({ query }: { query: string }) => search(query));   // each call recorded as a span

await context({ user_id: "u_123", session_id: "s_9" }, () => research(question));  // tags every run inside
```

- Tool and model calls are recorded as OpenTelemetry-shaped spans (`execute_tool webSearch`, `gen_ai.*` attributes), on `getActiveRun().events`.
- When a governed run throws, Fences records the error's type, message and stack with the run; the error still reaches your code.
- `init({ redact: (event) => ... })` sees everything before it leaves the process and can change or drop it.

## Fences Cloud

```ts
init({ apiKey: process.env.FENCES_API_KEY, endpoint: "https://your-fences-host", environment: "prod", release: "v1.4.2" });
```

The server is authoritative for limits and sees spend across processes. If it can't be reached, limits are enforced locally; pass `failClosed: true` to stop instead. A rejected key throws `FencesAuthError`; a quarantined agent throws `AgentQuarantined` before any of its code runs. In serverless handlers, `await flush()` before returning so queued decisions and spans are sent.

## Live control (from 0.4.0)

From version 0.4.0: in cloud mode a heartbeat (every 15 seconds, `init({ heartbeatS })`) lets whoever is on call act on a running agent from the dashboard:

- **Stop:** the next `checkpoint()` returns `breachType: "stopped_by_user"`, so the agent wraps up gracefully.
- **Pause and resume:** a paused run waits inside its next `checkpoint()` until it's resumed; after `pauseTimeoutS` (default one hour) it returns `breachType: "paused"`.
- **Limits** changed in the dashboard apply to the run immediately. **Quarantine** makes the governed call throw `AgentQuarantined` before any agent code runs.

An agent can ask a person before doing something risky:

```ts
import { requestApproval } from "agentfences";

const approval = await requestApproval("Refund $240 to order 1182?", { amountUsd: 240, timeoutS: 600 });
if (!approval.granted) return `I didn't issue the refund: ${approval.note}`;
```

An approved amount raises the run's budget. It never hangs or throws: in local mode, outside a run, or on a server without approvals it's denied right away, and with no answer it's denied after `timeoutS`. The heartbeat's timer never keeps your process alive on its own.

## Not yet in the TypeScript SDK

The Python SDK also records OpenAI/Anthropic calls automatically (`instrument`) and integrates with LangGraph and the OpenAI Agents SDK. Everything here follows the same server protocol ([PROTOCOL.md](../PROTOCOL.md)).
