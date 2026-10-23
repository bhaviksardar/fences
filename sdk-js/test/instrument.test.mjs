// instrument(): the real OpenAI and Anthropic SDKs against a fake model server, with budget prechecks.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import * as fences from "../dist/index.js";

// The openai package (7.x) needs Node 22+; instrument() itself doesn't care which Node or SDK version
const skip = Number(process.versions.node.split(".")[0]) < 22 && "the openai package used here needs Node 22+";

const { governed, checkpoint, init, getActiveRun, instrument, FencesStop } = fences;
const warnings = [];
console.warn = (m) => warnings.push(String(m));
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

// ── fake model server ─────────────────────────────────────────────────────────
const model = { requests: [], delayMs: 0, fail: false };
const fencesServer = { beats: [], commands: [] };  // just enough of a Fences server for onBreach "hold"
const usage = { prompt_tokens: 1000, completion_tokens: 200 };
let srv, base, openai, anthropic;

before(async () => {
  if (skip) return;
  const { default: OpenAI } = await import("openai");
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => raw += c);
    req.on("end", async () => {
      if (req.url.startsWith("/api/")) {
        const body = JSON.parse(raw || "{}");
        res.writeHead(200, { "content-type": "application/json" });
        if (req.url === "/api/heartbeat") {
          fencesServer.beats.push(body.runs);
          const commands = fencesServer.commands;
          fencesServer.commands = [];
          return res.end(JSON.stringify({ commands }));
        }
        return res.end(JSON.stringify({ ok: true }));
      }
      model.requests.push({ path: req.url, body: JSON.parse(raw || "{}") });
      await sleep(model.delayMs);
      const reply = (status, obj) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
      if (model.fail) return reply(500, { error: { message: "overloaded" } });
      if (req.url.endsWith("/chat/completions")) {
        return reply(200, { id: "c1", object: "chat.completion", created: 1, model: "gpt-4o-2024-08-06", usage,
          choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "hi" } }] });
      }
      if (req.url.endsWith("/responses")) {
        return reply(200, { id: "r1", object: "response", created_at: 1, model: "gpt-4o", status: "completed", output: [],
          usage: { input_tokens: 1000, output_tokens: 200, input_tokens_details: { cached_tokens: 0 } } });
      }
      reply(200, { id: "m1", type: "message", role: "assistant", model: "claude-sonnet-4-5", content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn", usage: { input_tokens: 1000, output_tokens: 200 } });
    });
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${srv.address().port}`;
  openai = instrument(new OpenAI({ apiKey: "sk-test", baseURL: `${base}/v1`, maxRetries: 0 }));
  anthropic = instrument(new Anthropic({ apiKey: "sk-test", baseURL: base, maxRetries: 0 }));
  init({ localOnly: true });
});
after(() => srv?.close());

const reset = () => Object.assign(model, { requests: [], delayMs: 0, fail: false });
const chat = (extra = {}) => openai.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "hi" }], max_tokens: 100, ...extra });
const gpt4o = fences.priceOf("gpt-4o");
const PER_CALL = 1000 * gpt4o.in + 200 * gpt4o.out;

test("model calls are recorded as chat spans with tokens and cost, and don't count as steps", { skip }, async () => {
  reset();
  assert.equal(instrument(openai), openai);  // again: nothing changes
  assert.equal((await chat()).choices[0].message.content, "hi");  // outside a run: just calls through
  const run = await governed(async () => {
    await chat();
    await openai.responses.create({ model: "gpt-4o", input: "hi", max_output_tokens: 100 });
    await anthropic.messages.create({ model: "claude-sonnet-4-5", max_tokens: 100, messages: [{ role: "user", content: "hi" }] });
    const { data, response } = await chat().withResponse();  // the SDK's own promise comes back, with its methods
    assert.equal(data.id, "c1");
    assert.equal(response.status, 200);
    return getActiveRun();
  }, { budgetUsd: 1 })();
  assert.deepEqual(run.events.map((s) => [s.name, s.attributes["gen_ai.provider.name"]]),
    [["chat gpt-4o-2024-08-06", "openai"], ["chat gpt-4o", "openai"], ["chat claude-sonnet-4-5", "anthropic"], ["chat gpt-4o-2024-08-06", "openai"]]);
  const a = run.events[0].attributes;
  assert.equal(a["gen_ai.usage.input_tokens"], 1000);
  assert.equal(a["gen_ai.usage.output_tokens"], 200);
  assert.ok(Math.abs(a["fences.cost_usd"] - PER_CALL) < 1e-12);
  assert.equal(run.iterations, 0);  // recorded, not checkpointed
});

test("a call whose worst case doesn't fit the budget is refused before it's sent", { skip }, async () => {
  reset();
  const outcome = await governed(async () => {
    try {
      await chat({ max_tokens: undefined });  // no cap: gpt-4o's whole 16,384-token output limit
      return "sent";
    } catch (e) {
      return e instanceof FencesStop ? e.breachType : String(e);
    }
  }, { budgetUsd: 0.01 })();
  assert.equal(outcome, "budget_exceeded");
  assert.equal(model.requests.length, 0, "the request never reached the provider");
  const refusedCall = await governed(async () => {
    const p = chat({ max_tokens: undefined });  // like the SDK's own: a promise, never a synchronous throw
    assert.ok(typeof p.then === "function");
    return p.withResponse().then(() => "sent", (e) => e.constructor.name);
  }, { budgetUsd: 0.01 })();
  assert.equal(refusedCall, "FencesStop");
  assert.ok(warnings.some((w) => /Setting max_tokens/.test(w)));
});

test("parallel calls can't overspend together: each holds its worst case while in flight", { skip }, async () => {
  reset();
  model.delayMs = 50;
  const results = await governed(async () => {
    const all = await Promise.allSettled(Array.from({ length: 5 }, () => chat()));
    return all.map((r) => r.status === "fulfilled" ? "sent" : r.reason.constructor.name);
  }, { budgetUsd: 0.0025 })();  // fits two calls capped at 100 output tokens (about $0.0011 each, worst case)
  assert.deepEqual(results.sort(), ["FencesStop", "FencesStop", "FencesStop", "sent", "sent"]);
  assert.equal(model.requests.length, 2);
});

test("a hold shrinks to the actual cost, and the checkpoint that records it releases it", { skip }, async () => {
  reset();
  const [during, after] = await governed(async () => {
    const response = await chat();
    const during = getActiveRun().reservedUsd;
    await checkpoint(response);
    return [during, [getActiveRun().reservedUsd, getActiveRun().costUsd]];
  }, { budgetUsd: 1 })();
  assert.ok(Math.abs(during - PER_CALL) < 1e-12, String(during));
  assert.ok(Math.abs(after[0]) < 1e-12 && Math.abs(after[1] - PER_CALL) < 1e-12, JSON.stringify(after));
});

test("a checkpoint releases only the call it prices: another finished call still counts until its own", { skip }, async () => {
  reset();
  const [counted, third] = await governed(async () => {
    const [r0, r1] = await Promise.all([chat(), chat()]);
    await checkpoint(r0);
    const run = getActiveRun(), counted = run.costUsd + run.reservedUsd;
    const third = await chat().then(() => "sent", (e) => e.constructor.name);
    await checkpoint(r1);
    return [counted, third];
  }, { budgetUsd: 0.0025 })();  // fits two calls capped at 100 output tokens
  assert.ok(Math.abs(counted - 2 * PER_CALL) < 1e-12, String(counted));
  assert.equal(third, "FencesStop");  // the second call was spent: a third could go over the budget
  assert.equal(model.requests.length, 2);
});

test("a call handled only with .catch() or .finally() is still recorded and settled", { skip }, async () => {
  reset();
  model.fail = true;
  const [held, spans] = await governed(async () => {
    await chat().catch(() => {});
    await chat().finally(() => {}).catch(() => {});
    return [getActiveRun().reservedUsd, getActiveRun().events.map((e) => e.status)];
  }, { budgetUsd: 1 })();
  assert.equal(held, 0);
  assert.deepEqual(spans, ["error", "error"]);
});

test("a failed call is released uncharged and recorded as an error", { skip }, async () => {
  reset();
  model.fail = true;
  const [err, held, span] = await governed(async () => {
    const err = await chat().then(() => null, (e) => e.status);
    return [err, getActiveRun().reservedUsd, getActiveRun().events.at(-1)];
  }, { budgetUsd: 1 })();
  assert.equal(err, 500);
  assert.equal(held, 0);
  assert.equal(span.status, "error");
  assert.equal(span.attributes["http.response.status_code"], 500);
});

test("asResponse(): the caller reads the body, so the call keeps its worst case until the next checkpoint", { skip }, async () => {
  reset();
  const [text, held, spans] = await governed(async () => {
    const raw = await chat().asResponse();
    const text = await raw.text();  // still readable: nothing read it first
    const held = getActiveRun().reservedUsd;
    await checkpoint(JSON.parse(text));
    return [text, [held, getActiveRun().reservedUsd], getActiveRun().events.length];
  }, { budgetUsd: 1 })();
  assert.equal(JSON.parse(text).id, "c1");
  assert.ok(held[0] > 0 && Math.abs(held[1]) < 1e-12, JSON.stringify(held));  // worst case held, then released
  assert.equal(spans, 1);  // recorded, without usage
});

test("a request the precheck can't serialize still goes out", { skip }, async () => {
  reset();
  const circular = { role: "user", content: "hi" };
  circular.self = circular;
  const sent = await governed(async () => {
    const p = chat({ metadata: { big: 10n } });  // a BigInt: JSON.stringify would throw
    assert.ok(typeof p.then === "function");
    await p.catch(() => {});  // the SDK itself may reject it; the precheck mustn't throw first
    return model.requests.length + getActiveRun().events.length;
  }, { budgetUsd: 1 })();
  assert.ok(sent >= 1, String(sent));
});

test("instrument() refuses what isn't a model client", { skip }, () => {
  assert.throws(() => instrument({}), TypeError);
});

test("onBreach hold: a call that doesn't fit waits for the budget to be raised, then is sent", { skip }, async () => {
  reset();
  init({ apiKey: "fc_test", endpoint: base, heartbeatS: 0.05 });
  try {
    const operator = (async () => {
      while (!fencesServer.beats.at(-1)?.some((r) => r.held === "budget")) await sleep(20);
      const runId = fencesServer.beats.at(-1)[0].run_id;
      fencesServer.commands = [{ run_id: runId, type: "limits", budget_usd: 1 }, { run_id: runId, type: "resume" }];
    })();
    const sent = await governed(async () => {
      const { data, response } = await chat().withResponse();  // the SDK promise's methods still work while it waits
      return [data.id, response.status, getActiveRun().events.length];
    }, { budgetUsd: 0.0005, onBreach: "hold" })();
    await operator;
    assert.deepEqual(sent, ["c1", 200, 1]);
    assert.equal(model.requests.length, 1);
  } finally {
    init({ localOnly: true });
  }
});
