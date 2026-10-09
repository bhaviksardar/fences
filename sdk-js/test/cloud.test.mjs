// Cloud mode against a fake Fences server: the real client, real fetch, real HTTP.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import * as fences from "../dist/index.js";

const { governed, checkpoint, init, getActiveRun, tool, context, logDecision, flush } = fences;
const warnings = [];
console.warn = (m) => warnings.push(String(m));

// ── fake server ───────────────────────────────────────────────────────────────
const server = { requests: [], budget: null, spent: 0, limits: null, notices: null, quarantine: false,
                 rejectKey: false, spansRoute: true };
const reset = () => Object.assign(server, { requests: [], budget: null, spent: 0, limits: null, notices: null,
                                             quarantine: false, rejectKey: false, spansRoute: true });
const sent = (path) => server.requests.filter((r) => r.path.endsWith(path));
let base, srv;

before(async () => {
  srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => raw += c);
    req.on("end", () => {
      const body = JSON.parse(raw || "{}");
      server.requests.push({ path: req.url, body, sdk: req.headers["x-fences-sdk"], key: req.headers["x-api-key"] });
      const reply = (status, obj) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
      if (server.rejectKey) return reply(401, { detail: "Invalid or revoked API key" });
      if (req.url === "/api/runs/start") {
        if (server.quarantine) return reply(423, { detail: "quarantined after 3 loops" });
        server.spent = 0;
        const r = { ok: true };
        if (server.limits) r.limits = server.limits;
        if (server.notices) r.notices = server.notices;
        return reply(200, r);
      }
      if (req.url.endsWith("/checkpoint")) {
        server.spent += body.cost_delta_usd;
        if (server.budget !== null && Math.round(server.spent * 1e9) / 1e9 > server.budget) {
          return reply(200, { ok: false, breach: "budget_exceeded", spent_usd: server.spent, budget_usd: server.budget });
        }
        return reply(200, { ok: true, spent_usd: server.spent });
      }
      if (req.url.endsWith("/spans") && !server.spansRoute) return reply(404, { detail: "Not Found" });
      reply(200, { ok: true });
    });
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${srv.address().port}`;
});
after(() => srv.close());

const cloud = (opts = {}) => { reset(); init({ apiKey: "fc_test", endpoint: base, ...opts }); };

// ── tests ─────────────────────────────────────────────────────────────────────
test("a run's start, checkpoints, decisions and end, with the version header", async () => {
  cloud();
  const agent = governed(async function researchAgent() {
    logDecision("searching", "search");
    await checkpoint({ model: "gpt-4o", usage: { prompt_tokens: 1000, completion_tokens: 200 } });
    return "done";
  }, { budgetUsd: 2, maxIterations: 20 });
  assert.equal(await agent(), "done");
  await flush();
  const [start] = sent("/api/runs/start");
  assert.equal(start.body.agent_name, "researchAgent");
  assert.deepEqual([start.body.budget_usd, start.body.max_iterations, start.body.max_duration_ms, start.body.max_tokens], [2, 20, null, null]);
  assert.ok(server.requests.every((r) => r.sdk === `js/${fences.VERSION}` && r.key === "fc_test"));
  const [cp] = sent("/checkpoint");
  assert.ok(cp.body.cost_delta_usd > 0 && cp.body.tokens_used === 1200 && cp.body.iterations === 1);
  assert.deepEqual(sent("/decisions")[0].body, { iteration: 0, reasoning: "searching", action: "search" });
  assert.deepEqual(sent("/end")[0].body, { status: "success", error: null });
});

test("the server decides breaches", async () => {
  cloud();
  server.budget = 0.05;
  const r = await governed(async () => {
    for (;;) {
      const x = await checkpoint(null, { costDeltaUsd: 0.02 });
      if (x.breached) return x;
    }
  }, { budgetUsd: 99 })();  // local says fine; the server knows better
  assert.equal(r.breachType, "budget_exceeded");
  assert.ok(Math.abs(r.spentUsd - 0.06) < 1e-9);
  assert.equal(sent("/end")[0].body.status, "breached");
});

test("server limits and notices are adopted at run start", async () => {
  cloud();
  server.limits = { budget_usd: null, max_iterations: 3, max_duration_ms: 300000, max_tokens: 0 };
  server.notices = [{ code: "no_budget", message: "freeAgent has no budget, so its spend isn't capped.", url: "https://fences.example/agents/freeAgent" }];
  warnings.length = 0;
  const agent = governed(async function freeAgent() { const r = getActiveRun(); return [r.budgetUsd, r.maxIterations]; });
  assert.deepEqual(await agent(), [null, 3]);
  await agent();
  assert.equal(warnings.filter((w) => w === "agentfences: freeAgent has no budget, so its spend isn't capped. https://fences.example/agents/freeAgent").length, 1);
});

test("errors end the run with the exception", async () => {
  cloud();
  function fetchPage() { throw new TypeError("page 7 returned 403"); }
  await assert.rejects(governed(async () => fetchPage(), { budgetUsd: 1 })(), /page 7/);
  const end = sent("/end")[0].body;
  assert.equal(end.status, "error");
  assert.equal(end.error, "TypeError: page 7 returned 403");
  assert.equal(end.exception.type, "TypeError");
  assert.ok(end.exception.stack.at(-1).endsWith("in fetchPage"), end.exception.stack.at(-1));
});

test("quarantine, bad keys and an unreachable server", async () => {
  cloud();
  server.quarantine = true;
  let ran = false;
  await assert.rejects(governed(async () => { ran = true; }, { budgetUsd: 1 })(), (e) =>
    e instanceof fences.AgentQuarantined && /quarantined after 3 loops/.test(e.message));
  assert.ok(!ran, "no agent code runs");

  cloud();
  server.rejectKey = true;
  await assert.rejects(governed(async () => 1, { budgetUsd: 1 })(), fences.FencesAuthError);

  init({ apiKey: "fc_test", endpoint: "http://127.0.0.1:9" });  // nothing listens there
  warnings.length = 0;
  const r = await governed(async () => {
    for (;;) {
      const x = await checkpoint(null, { costDeltaUsd: 0.02 });
      if (x.breached) return x.breachType;
    }
  }, { budgetUsd: 0.05 })();
  assert.equal(r, "budget_exceeded", "limits are enforced locally");
  assert.equal(warnings.filter((w) => w.includes("unreachable")).length, 1);

  init({ apiKey: "fc_test", endpoint: "http://127.0.0.1:9", failClosed: true });
  const closed = await governed(async () => (await checkpoint()).breachType, { budgetUsd: 1 })();
  assert.equal(closed, "fences_unreachable");
});

test("spans are sent in batches, redacted; an older server without /spans gets one warning", async () => {
  cloud({ redact: (e) => {
    if (e.type !== "span") return e;
    if (e.attributes["gen_ai.tool.name"] === "secret") return null;
    e.attributes["gen_ai.tool.call.arguments"] = e.attributes["gen_ai.tool.call.arguments"].replace("hunter2", "***");
    return e;
  } });
  const login = tool(function login(password) { return true; });
  const secret = tool(function secret() { return 1; });
  const n = await governed(async () => {
    for (let i = 0; i < 30; i++) login("hunter2");
    secret();
    return getActiveRun().events.length;
  }, { budgetUsd: 1 })();
  assert.equal(n, 31, "everything is kept on the run locally");
  await flush();
  const batches = sent("/spans");
  const spans = batches.flatMap((b) => b.body.spans);
  assert.equal(spans.length, 30);
  assert.ok(batches.length < 30, `batched: ${batches.length} requests`);
  assert.ok(spans.every((s) => s.attributes["gen_ai.tool.call.arguments"] === '"***"' && !("type" in s)));
  assert.ok(!JSON.stringify(server.requests).includes("hunter2"));

  cloud();
  server.spansRoute = false;
  warnings.length = 0;
  const ping = tool(function ping() { return "pong"; });
  await governed(async () => { for (let i = 0; i < 5; i++) { ping(); await flush(); } }, { budgetUsd: 1 })();
  assert.equal(warnings.filter((w) => w.includes("doesn't accept tool and model call spans")).length, 1);
});

test("context and redaction on run start and end", async () => {
  cloud({ environment: "prod", redact: (e) => {
    if (e.type === "run_start") delete e.context.user_id;
    if (e.type === "run_end") return { ...e, error: "redacted", exception: null };
    return e;
  } });
  await context({ user_id: "alice", session_id: "s_1" }, () =>
    assert.rejects(governed(async () => { throw new Error("SMTP rejected alice@example.com"); }, { budgetUsd: 1 })()));
  assert.deepEqual(sent("/api/runs/start")[0].body.context, { environment: "prod", session_id: "s_1" });
  assert.deepEqual(sent("/end")[0].body, { status: "error", error: "redacted" });
  assert.ok(!JSON.stringify(server.requests).includes("alice"));
});
