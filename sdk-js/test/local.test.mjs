// Local mode: no server, no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as fences from "../dist/index.js";
import { argsSummary } from "../dist/tools.js";

const { governed, checkpoint, init, getActiveRun, tool, context, priceOf } = fences;
const close = (a, b) => Math.abs(a - b) <= 1e-12 + 1e-9 * Math.abs(b);
const warnings = [];
console.warn = (m) => warnings.push(String(m));

test("a governed call before init() throws", async () => {
  await assert.rejects(governed(async () => 1, { budgetUsd: 1 })(), /Call agentfences.init\(\)/);
  init({ localOnly: true });
});

test("README quickstart: stops at $0.12 of a $0.10 budget with the right message", async () => {
  const agent = governed(async () => {
    for (let step = 0; step < 100; step++) {
      fences.logDecision(`step ${step}: searching`, "search");
      const r = await checkpoint(null, { costDeltaUsd: 0.02, tokensUsed: 200 });
      if (r.breached) return r.message;
    }
    return "done";
  }, { budgetUsd: 0.1, maxIterations: 20 });
  assert.equal(await agent(), "I've reached my budget limit (spent $0.1200 of $0.1000). I'll summarize what I found so far.");
});

async function stepsUntilBreach(limits) {
  return governed(async () => {
    for (let step = 0; step < 1000; step++) {
      const r = await checkpoint(null, { costDeltaUsd: 0.02, tokensUsed: 100 });
      if (r.breached) return [step, r];
    }
  }, limits)();
}

test("limits trip once exceeded, not when reached", async () => {
  let [steps, r] = await stepsUntilBreach({ budgetUsd: 0.1 });
  assert.deepEqual([steps, r.breachType], [5, "budget_exceeded"]);
  [steps, r] = await stepsUntilBreach({ budgetUsd: 99, maxIterations: 5 });
  assert.deepEqual([steps, r.breachType], [5, "iteration_limit"]);
  [steps, r] = await stepsUntilBreach({ budgetUsd: 99, maxTokens: 300 });
  assert.deepEqual([steps, r.breachType], [3, "token_limit"]);
  assert.ok(r.breached && !r.ok && r.systemPrompt);
  assert.equal(r.message, "I've reached my token limit (400 tokens). I'll summarize what I found so far.");
});

test("time limit", async () => {
  const r = await governed(async () => {
    for (;;) {
      await new Promise((ok) => setTimeout(ok, 30));
      const x = await checkpoint();
      if (x.breached) return x.breachType;
    }
  }, { budgetUsd: 99, maxDurationMs: 100 })();
  assert.equal(r, "time_limit");
});

test("concurrent runs are independent, nested runs are separate", async () => {
  const agent = governed(async () => {
    for (let step = 0; step < 100; step++) {
      await new Promise((ok) => setImmediate(ok));
      if ((await checkpoint(null, { costDeltaUsd: 0.02 })).breached) return step;
    }
  }, { budgetUsd: 0.1 });
  assert.deepEqual(await Promise.all([agent(), agent(), agent()]), [5, 5, 5]);

  const inner = governed(async () => { await checkpoint(); return getActiveRun().runId; }, { budgetUsd: 1 });
  const outer = governed(async () => {
    const id = getActiveRun().runId, innerId = await inner();
    assert.equal(getActiveRun().runId, id);
    assert.notEqual(innerId, id);
    await checkpoint();
    return getActiveRun().iterations;
  }, { budgetUsd: 1 });
  assert.equal(await outer(), 1);  // the inner run's checkpoint didn't count here
  assert.equal(getActiveRun(), null);
  assert.ok((await checkpoint(null, { costDeltaUsd: 5 })).ok, "outside a run: nothing to check");
});

test("a number as the first argument is rejected, not priced", async () => {
  await assert.rejects(governed(async () => checkpoint(0.02), { budgetUsd: 1 })(), TypeError);
});

test("no budget: runs unlimited, warns once per agent", async () => {
  warnings.length = 0;
  const unlimitedAgent = governed(async function unlimitedAgent() {
    for (let i = 0; i < 50; i++) assert.ok((await checkpoint(null, { costDeltaUsd: 1000 })).ok);
    const r = getActiveRun();
    return [r.budgetUsd, r.maxIterations, r.maxDurationMs, r.maxTokens];
  });
  assert.deepEqual(await unlimitedAgent(), [null, 100, 300000, 0]);
  await unlimitedAgent();
  assert.equal(warnings.filter((w) => w.includes("unlimitedAgent has no budget")).length, 1);
});

function priced(model, { inp = 0, read = 0, write = 0, out = 0 }) {
  let p = priceOf(model);
  assert.ok(p, `${model} missing from prices`);
  if (p.above !== undefined && inp + read + write > p.above) p = { ...p, ...p.tier };
  return inp * p.in + read * (p.cache_read ?? p.in) + write * (p.cache_write ?? p.in) + out * p.out;
}
const step = (response, extra) => governed(async () => {
  const r = await checkpoint(response, extra);
  return [r.spentUsd, getActiveRun().tokensUsed];
}, { budgetUsd: 1e9 })();

test("prices each response shape", async () => {
  let [spent, tokens] = await step({ model: "gpt-4o-2024-08-06", usage: { prompt_tokens: 1000, completion_tokens: 200, prompt_tokens_details: { cached_tokens: 400 } } });
  assert.ok(close(spent, priced("gpt-4o", { inp: 600, read: 400, out: 200 })) && tokens === 1200, "OpenAI chat");
  [spent] = await step({ model: "gpt-4o", usage: { input_tokens: 1000, output_tokens: 200, input_tokens_details: { cached_tokens: 300, cache_write_tokens: 500 } } });
  assert.ok(close(spent, priced("gpt-4o", { inp: 200, read: 300, write: 500, out: 200 })), "OpenAI Responses");
  [spent, tokens] = await step({ model: "claude-sonnet-4-5-20250929", usage: { input_tokens: 100, output_tokens: 50, cache_read_input_tokens: 1000, cache_creation_input_tokens: 500 } });
  assert.ok(close(spent, priced("claude-sonnet-4-5", { inp: 100, read: 1000, write: 500, out: 50 })) && tokens === 1650, "Anthropic");
  [spent] = await step({ model: "claude-sonnet-4-5", usage: { input_tokens: 250000, output_tokens: 1000 } });
  assert.ok(spent > priced("claude-sonnet-4-5", { inp: 200000, out: 1000 }) * 1.25, "long-context tier");
  [spent, tokens] = await step({ modelVersion: "gemini-2.5-flash", usageMetadata: { promptTokenCount: 1000, candidatesTokenCount: 100, thoughtsTokenCount: 300 } });
  assert.ok(close(spent, priced("gemini-2.5-flash", { inp: 1000, out: 400 })) && tokens === 1400, "Gemini (thinking billed as output)");
  [spent, tokens] = await step({ response: { modelId: "gpt-4o" }, usage: { inputTokens: 1000, outputTokens: 200, totalTokens: 1200, cachedInputTokens: 400 } });
  assert.ok(close(spent, priced("gpt-4o", { inp: 600, read: 400, out: 200 })) && tokens === 1200, "Vercel AI SDK");
  [spent] = await step({ response_metadata: { model_name: "gpt-4o" }, usage_metadata: { input_tokens: 1000, output_tokens: 100, input_token_details: { cache_read: 200 } } });
  assert.ok(close(spent, priced("gpt-4o", { inp: 800, read: 200, out: 100 })), "LangChain.js");
  assert.deepEqual(await step({ model: "anything", usage: { prompt_tokens: 10, completion_tokens: 5, cost: 0.0123 } }), [0.0123, 15]);
  [spent, tokens] = await step({ model: "gpt-4o", usage: { prompt_tokens: 1000, completion_tokens: 0 } }, { costDeltaUsd: 0.01, tokensUsed: 5 });
  assert.ok(close(spent, priced("gpt-4o", { inp: 1000 }) + 0.01) && tokens === 1005, "extra costs add up");
});

test("unknown models are counted and warned about; custom prices fix them", async () => {
  warnings.length = 0;
  const r = { model: "my-finetune-v2", usage: { prompt_tokens: 1e6, completion_tokens: 1e6 } };
  assert.deepEqual(await step(r), [0, 2e6]);
  await step(r);
  assert.equal(warnings.filter((w) => w.includes("no price for model 'my-finetune-v2'")).length, 1);
  init({ localOnly: true, prices: { "my-finetune-v2": { input: 2, output: 6 } } });
  assert.ok(close((await step(r))[0], 8));
  init({ localOnly: true });
});

test("tool() records calls as execute_tool spans", async () => {
  const webSearch = tool(function webSearch({ query, limit = 5 }) {
    if (query === "blocked") throw Object.assign(new Error("403 from search API"), { name: "PermissionError" });
    return Array(limit).fill(query);
  });
  const fetchPage = tool(async (url) => { await new Promise((ok) => setTimeout(ok, 10)); return "<html>"; }, { name: "fetch" });
  assert.deepEqual(webSearch({ query: "outside", limit: 1 }), ["outside"]);  // no run: just calls through, stays sync

  const spans = await governed(async () => {
    assert.equal(webSearch({ query: "cats", limit: 2 }).length, 2);  // sync tools stay sync
    await fetchPage("https://example.com/" + "x".repeat(500));
    assert.throws(() => webSearch({ query: "blocked" }), /403/);
    return getActiveRun().events;
  }, { budgetUsd: 1 })();
  assert.deepEqual(spans.map((s) => [s.name, s.status]),
    [["execute_tool webSearch", "ok"], ["execute_tool fetch", "ok"], ["execute_tool webSearch", "error"]]);
  const a = spans.map((s) => s.attributes);
  assert.equal(a[0]["gen_ai.tool.call.arguments"], '{"query":"cats","limit":2}');
  assert.equal(a[0]["gen_ai.operation.name"], "execute_tool");
  assert.ok(a[1]["gen_ai.tool.call.arguments"].length <= 300 && spans[1].end_ts - spans[1].start_ts >= 0.009);
  assert.equal(a[2]["error.type"], "PermissionError");
  assert.equal(a[2]["fences.error.message"], "PermissionError: 403 from search API");
  assert.equal(argsSummary([1, "two"]), '[1,"two"]');
  const loop = {}; loop.self = loop;
  assert.equal(argsSummary([loop]), "[object Object]");
});

test("context() tags runs; inner values win", async () => {
  init({ localOnly: true, environment: "prod", release: "v1.4.2" });
  const agent = governed(async () => ({ ...getActiveRun().context }), { budgetUsd: 1 });
  const [inner, outer] = await context({ user_id: "u_1", session_id: "s_1" }, async () => [
    await context({ session_id: "s_2", trace_id: null }, agent), await agent()]);
  assert.deepEqual(inner, { environment: "prod", release: "v1.4.2", user_id: "u_1", session_id: "s_2" });
  assert.deepEqual(outer, { environment: "prod", release: "v1.4.2", user_id: "u_1", session_id: "s_1" });
  assert.deepEqual(await agent(), { environment: "prod", release: "v1.4.2" });
  init({ localOnly: true });
});
