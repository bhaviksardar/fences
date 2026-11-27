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
                 rejectKey: false, spansRoute: true, revokeAfterStart: false, checkpointReplies: [], refuseApprovals: false, echoSteps: false, dropNext: 0 };
const reset = () => Object.assign(server, { requests: [], budget: null, spent: 0, limits: null, notices: null,
                                             quarantine: false, rejectKey: false, spansRoute: true, revokeAfterStart: false,
                                             checkpointReplies: [], refuseApprovals: false, echoSteps: false, dropNext: 0 });
const sent = (path) => server.requests.filter((r) => r.path.endsWith(path));
let base, srv;

before(async () => {
  srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => raw += c);
    req.on("end", () => {
      const body = JSON.parse(raw || "{}");
      if (server.redirectTo && req.url.endsWith("/checkpoint")) {
        res.writeHead(307, { location: server.redirectTo });
        return res.end();
      }
      if (server.dropNext && req.url.endsWith("/checkpoint")) {  // the connection drops before any reply
        server.dropNext -= 1;
        server.requests.push({ path: req.url, body, dropped: true });
        return req.socket.destroy();
      }
      server.requests.push({ path: req.url, body, sdk: req.headers["x-fences-sdk"], key: req.headers["x-api-key"] });
      const reply = (status, obj) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
      if (server.rejectKey || (server.revokeAfterStart && req.url !== "/api/runs/start")) {
        return reply(401, { detail: "Invalid or revoked API key" });
      }
      if (req.url === "/api/runs/start") {
        if (server.quarantine) return reply(423, { detail: "quarantined after 3 loops" });
        server.spent = 0;
        const r = { ok: true };
        if (server.limits) r.limits = server.limits;
        if (server.notices) r.notices = server.notices;
        return reply(200, r);
      }
      if (req.url.endsWith("/checkpoint") && server.checkpointReplies.length) {
        const [status, obj] = server.checkpointReplies.shift();
        return reply(status, obj);
      }
      if (req.url.endsWith("/approvals") && server.refuseApprovals) return reply(422, { detail: "bad request" });
      if (req.url.endsWith("/checkpoint")) {
        server.spent += body.cost_delta_usd;
        if (server.budget !== null && Math.round(server.spent * 1e9) / 1e9 > server.budget) {
          return reply(200, { ok: false, breach: "budget_exceeded", spent_usd: server.spent, budget_usd: server.budget });
        }
        return reply(200, { ok: true, spent_usd: server.spent, ...(server.echoSteps && body.step_id ? { step_id: body.step_id } : {}) });
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
  cloud({ toolArguments: "full", redact: (e) => {  // values, so the redact hook has something to scrub
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

  server.spansRoute = true;
  cloud();  // init() asks the new server again
  const before = sent("/spans").length;
  await governed(async () => { ping(); await flush(); }, { budgetUsd: 1 })();
  assert.equal(sent("/spans").length, before + 1);
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

test("a key revoked mid-run ends the run at its next checkpoint; the agent's result still comes back", async () => {
  cloud();
  server.revokeAfterStart = true;
  const results = await governed(async () => [await checkpoint(null, { costDeltaUsd: 0.01 }), "agent result"], { budgetUsd: 1 })();
  assert.equal(results[0].breachType, "fences_unreachable");
  assert.equal(results[1], "agent result");  // endRun's 401 doesn't throw over it
  await assert.rejects(governed(async () => { throw new TypeError("boom"); }, { budgetUsd: 1 })(), TypeError);  // its own error, not FencesAuthError
});

test("null or odd numbers in a reply never break local enforcement later", async () => {
  cloud();
  server.checkpointReplies = [[200, { ok: true, spent_usd: null, iterations: "3", max_iterations: null, budget_usd: null }],
                              [500, { detail: "oops" }], [200, ["not", "a", "reply"]]];
  const [a, b, c] = await governed(async () => [await checkpoint(null, { costDeltaUsd: 0.1 }),
    await checkpoint(null, { costDeltaUsd: 0.1 }), await checkpoint(null, { costDeltaUsd: 0.1 })], { budgetUsd: 1 })();
  assert.ok(a.ok && b.ok && c.ok, JSON.stringify([a, b, c]));  // unlimited (null budget); no false iteration_limit
});

test("an approval request the server refuses is denied at once, not after the timeout", async () => {
  cloud();
  server.refuseApprovals = true;
  const started = Date.now();
  const answer = await governed(async () => fences.requestApproval("Refund?", { timeoutS: 5 }), { budgetUsd: 1 })();
  assert.ok(!answer.granted && /HTTP 422/.test(answer.note) && Date.now() - started < 1000, JSON.stringify(answer));
});

test("every checkpoint has its own step_id; a dropped one is sent once more once the server dedupes", async () => {
  cloud();
  server.dropNext = 1;  // an old server (no echo): the dropped step isn't resent
  await governed(async () => { await checkpoint(null, { costDeltaUsd: 0.01 }); await checkpoint(null, { costDeltaUsd: 0.01 }); }, { budgetUsd: 1 })();
  let steps = sent("/checkpoint").map((r) => r.body.step_id);
  assert.equal(steps.length, 2);
  assert.ok(steps.every((id) => /^[0-9a-f]{32}$/.test(id)) && steps[0] !== steps[1], JSON.stringify(steps));

  cloud();
  server.echoSteps = true;
  await governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.01 });  // echoed: this server dedupes
    server.dropNext = 1;
    const r = await checkpoint(null, { costDeltaUsd: 0.01 });  // dropped, then sent again with the same step_id
    assert.ok(r.ok && !r.breached);
  }, { budgetUsd: 1 })();
  steps = sent("/checkpoint").map((r) => r.body.step_id);
  assert.equal(steps.length, 3);
  assert.equal(steps[1], steps[2]);
});

test("a remote endpoint must use HTTPS; a reply can't make a budget set in code unlimited", async () => {
  for (const ok of ["https://fences.example.com", "http://localhost:8000", "http://127.0.0.1:9", "http://[::1]:8000", "http://LOCALHOST"]) {
    init({ apiKey: "fc_test", endpoint: ok });
  }
  for (const bad of ["http://fences.example.com", "http://10.0.0.5:8000", "ftp://fences.example.com", "fences.example.com",
                     "http://evil.com\\@localhost", "http://localhost@evil.com", "https://user:pw@fences.example.com", "http://evil.localhost"]) {
    assert.throws(() => init({ apiKey: "fc_test", endpoint: bad }), (e) => e instanceof TypeError && /https/.test(e.message), bad);
  }
  init({ apiKey: "fc_test", endpoint: "http://10.0.0.5:8000", allowHttp: true });  // a trusted private network

  cloud();
  server.checkpointReplies = [[200, { ok: true, budget_usd: null }], [200, { ok: true, budget_usd: 5 }]];
  const capped = await governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.1 });
    const afterNull = getActiveRun().budgetUsd;
    await checkpoint(null, { costDeltaUsd: 0.1 });
    return [afterNull, getActiveRun().budgetUsd];
  }, { budgetUsd: 1 })();
  assert.deepEqual(capped, [1, 5]);  // null ignored; a raise from the dashboard still applies

  server.limits = { budget_usd: 2 };
  server.checkpointReplies = [[200, { ok: true, budget_usd: null }]];
  const uncapped = await governed(async () => { await checkpoint(null, { costDeltaUsd: 0.1 }); return getActiveRun().budgetUsd; })();
  assert.equal(uncapped, null);  // no budget in code: the server's null (unlimited) stands
});

test("a redirect isn't followed: it would carry the API key to another host", async () => {
  const stolen = [];
  const attacker = http.createServer((req, res) => { stolen.push(req.headers["x-api-key"]); res.end("{}"); });
  await new Promise((ok) => attacker.listen(0, "127.0.0.1", ok));
  try {
    cloud();
    server.redirectTo = `http://127.0.0.1:${attacker.address().port}/steal`;
    const result = await governed(async () => checkpoint(null, { costDeltaUsd: 0.01 }), { budgetUsd: 1 })();
    assert.ok(result.ok);  // enforced locally, as with no answer
    assert.deepEqual(stolen, []);
    assert.ok(warnings.some((w) => /redirect \(HTTP 307/.test(w)), warnings.join("\n"));
  } finally {
    server.redirectTo = null;
    attacker.close();
  }
});

test("malformed replies can't break a checkpoint or a run start, lift a limit, or clear a breach", async () => {
  cloud();
  server.notices = [null, 1];
  server.checkpointReplies = [[200, { ok: false, breach: ["x"], commands: [null, 5, { type: "stop", id: ["x"] }] }],
                              [200, { ok: "false", breach: "budget_exceeded" }]];
  const breaches = await governed(async () => [(await checkpoint(null, { costDeltaUsd: 0.01 })).breachType,
                                               (await checkpoint(null, { costDeltaUsd: 0.01 })).breachType], { budgetUsd: 1 })();
  assert.deepEqual(breaches, ["limit_reached", "budget_exceeded"]);
  const { adopt, LIMITS } = await import("../dist/control.js");
  const run = { budgetUsd: 1 };
  adopt(run, { budget_usd: Infinity }, LIMITS);  // a reply body of 1e400 parses to Infinity
  assert.equal(run.budgetUsd, 1);
});

test("failed calls carry an error category; a failed run sends one in a fences.run.error span", async () => {
  cloud();
  const brokenTool = tool(function brokenTool() { throw new Error("bad row"); });
  const slowTool = tool(async function slowTool() { throw Object.assign(new Error("slow"), { code: "ETIMEDOUT" }); });
  const { recordSpan } = await import("../dist/core.js");
  class RateLimitError extends Error { status = 429; }
  class APIError extends Error { status = 529; }
  await assert.rejects(governed(async () => {
    assert.throws(() => brokenTool());
    await assert.rejects(slowTool());
    recordSpan("chat m", 0, { "gen_ai.operation.name": "chat" }, new Error("odd"));
    recordSpan("chat m", 0, { "gen_ai.operation.name": "chat" }, new RateLimitError("slow down"));
    throw new APIError("overloaded");
  }, { budgetUsd: 1 })());
  await flush();
  const spans = sent("/spans").flatMap((b) => b.body.spans);
  assert.deepEqual(spans.map((s) => [s.name, s.attributes["fences.error.category"]]), [
    ["execute_tool brokenTool", "tool_error"], ["execute_tool slowTool", "timeout"], ["chat m", "model_error"],
    ["chat m", "rate_limit"], ["fences.run.error", "overloaded"]]);
  const runError = spans.at(-1);
  assert.equal(runError.status, "error");
  assert.equal(runError.attributes["error.type"], "APIError");
  assert.equal(runError.attributes["http.response.status_code"], 529);
  assert.ok(!("fences.error.message" in runError.attributes), "the message is on the run end only");
  assert.ok(!("gen_ai.operation.name" in runError.attributes), "not a tool or model call");
  assert.equal(spans[3].attributes["http.response.status_code"], 429);

  await governed(async () => 1, { budgetUsd: 1 })();
  await flush();
  assert.equal(sent("/spans").flatMap((b) => b.body.spans).length, 5, "a run that succeeds has no run error span");
});

test("capture metadata: nothing the agent or its users wrote leaves the process, in any event", async () => {
  const secrets = { prompt: "patient Jane Roe has stage 2 diabetes", decision: "Jane looks worse; escalating", action: "page_dr_who",
    context: "mrn-0042", toolError: "record mrn-0042 locked by Dr Who", modelError: "prompt rejected: Jane Roe",
    runError: "chart for Jane Roe missing", label: "Jane asked for it", approval: "refund Jane Roe", hook: "could not revoke Jane's token" };
  const seen = [];
  cloud({ capture: "metadata", environment: "prod", redact: (e) => { seen.push(JSON.stringify(e)); return e; },
          onStop: () => { throw new Error(secrets.hook); } });
  const { recordSpan } = await import("../dist/core.js");
  const { recordLlmCall } = await import("../dist/spans.js");
  const control = await import("../dist/control.js");
  class RateLimitError extends Error { status = 429; }
  const lookup = tool(function lookup(question) { return "ok"; });
  const locked = tool(function locked(chartId) { throw new Error(secrets.toolError); });
  await context({ user_id: secrets.context }, () => assert.rejects(governed(async () => {
    lookup(secrets.prompt);
    assert.throws(() => locked("mrn-0042"));
    logDecision(secrets.decision, secrets.action);
    recordLlmCall("openai", "gpt-4o", performance.now(), undefined, new RateLimitError(secrets.modelError));
    fences.fallback("gpt-4o", "gpt-4o-mini", secrets.label);
    fences.fallback("gpt-4o", "gpt-4o-mini", new RateLimitError(secrets.modelError));
    recordSpan("fences.approval", 0, { "fences.approval.reason": secrets.approval, "fences.approval.note": secrets.approval,
                                       "fences.approval.granted": false });
    throw new Error(secrets.runError);
  }, { budgetUsd: 1 })()));
  await governed(async () => { control.apply(getActiveRun(), [{ type: "stop", id: "c1", run_id: getActiveRun().runId }]); }, { budgetUsd: 1 })();
  await flush();
  const out = JSON.stringify(server.requests) + seen.join("");
  assert.deepEqual(Object.keys(secrets).filter((k) => out.includes(secrets[k])), []);
  for (const name of ["Jane", "mrn-0042", "Dr Who", "page_dr"]) assert.ok(!out.includes(name), name);

  // What's still sent: the keys, names, models, numbers, error types, categories and statuses
  assert.deepEqual(sent("/api/runs/start")[0].body.context, { environment: "prod", user_id: "" });
  assert.equal(sent("/decisions").length, 0);
  const spans = Object.fromEntries(sent("/spans").flatMap((b) => b.body.spans).map((s) => [s.name, s]));
  assert.equal(spans["fences.decision"].attributes["fences.decision.length"], secrets.decision.length);
  assert.ok(!("gen_ai.tool.call.arguments" in spans["execute_tool lookup"].attributes));
  const t = spans["execute_tool locked"].attributes;
  assert.deepEqual([t["error.type"], t["fences.error.category"], "fences.error.message" in t], ["Error", "tool_error", false]);
  const chat = spans["chat gpt-4o"].attributes;
  assert.deepEqual([chat["fences.error.category"], chat["http.response.status_code"]], ["rate_limit", 429]);
  const fallbacks = sent("/spans").flatMap((b) => b.body.spans).filter((s) => s.name.startsWith("fallback"));
  assert.deepEqual(fallbacks.map((s) => s.attributes["fences.fallback.reason"]), [undefined, "rate_limit"], "a category stays, a label goes");
  assert.equal(spans["fences.approval"].attributes["fences.approval.granted"], false);
  const end = sent("/end")[0].body;
  assert.deepEqual([end.status, end.error, end.exception.type, end.exception.message], ["error", "Error", "Error", ""]);
  assert.ok(end.exception.stack.length, "where it failed is code, not data: kept");
  const acks = sent("/api/heartbeat").flatMap((r) => r.body.acks ?? []);
  assert.deepEqual(acks.filter((a) => a.id === "c1" && a.hook).map((a) => a.detail), ["Error"]);
  assert.throws(() => init({ localOnly: true, capture: "everything" }), /capture/);
});

test("tags: checked at init and governed(), merged (governed's win), sent in the run start's context as tag.<key>", async () => {
  const bad = [Object.fromEntries([...Array(21).keys()].map((i) => [`k${i}`, "v"])), { Team: "ops" }, { "team ops": "x" },
    { ["k".repeat(41)]: "v" }, { "": "v" }, { team: "x".repeat(101) }, { team: 7 }, { team: null }, ["team"], "team=ops"];
  for (const tags of bad) {
    assert.throws(() => init({ localOnly: true, tags }), undefined, JSON.stringify(tags));
    assert.throws(() => governed(async () => 1, { tags }), undefined, JSON.stringify(tags));
  }
  init({ localOnly: true, tags: Object.fromEntries([...Array(20).keys()].map((i) => [`k${i}`, "v".repeat(100)])) });  // at the limits: fine

  cloud({ tags: { pipeline: "pcr", team: "ops" }, environment: "prod" });
  const agent = governed(async () => [{ ...getActiveRun().tags }, { ...getActiveRun().context }],
                         { budgetUsd: 1, tags: { team: "billing", "stage.v2": "triage" } });
  const plain = governed(async () => ({ ...getActiveRun().tags }), { budgetUsd: 1 });
  const [tags, ctx] = await agent();
  assert.deepEqual(tags, { pipeline: "pcr", team: "billing", "stage.v2": "triage" }, "governed's win");
  assert.deepEqual(ctx, { environment: "prod" }, "tags aren't context; they're sent with it");
  assert.deepEqual(await plain(), { pipeline: "pcr", team: "ops" });
  assert.deepEqual(sent("/api/runs/start").map((r) => r.body.context), [
    { environment: "prod", "tag.pipeline": "pcr", "tag.team": "billing", "tag.stage.v2": "triage" },
    { environment: "prod", "tag.pipeline": "pcr", "tag.team": "ops" }]);

  // Tags are metadata: capture "metadata" keeps them, and blanks the context values around them
  cloud({ tags: { pipeline: "pcr" }, capture: "metadata" });
  await context({ user_id: "u_1" }, () => plain());
  assert.deepEqual(sent("/api/runs/start")[0].body.context, { user_id: "", "tag.pipeline": "pcr" });

  // A server takes 32 context keys: past that, environment, release and tags go first, with one warning
  cloud({ tags: Object.fromEntries([...Array(20).keys()].map((i) => [`t${i}`, "v"])), environment: "prod", release: "v1" });
  warnings.length = 0;
  const crowd = Object.fromEntries([...Array(30).keys()].map((i) => [`c${String(i).padStart(2, "0")}`, "x"]));
  await context(crowd, async () => { await plain(); await plain(); });
  const sentCtx = sent("/api/runs/start").at(-1).body.context;
  assert.equal(Object.keys(sentCtx).length, 32);
  assert.deepEqual([sentCtx.environment, sentCtx.release], ["prod", "v1"]);
  assert.ok([...Array(20).keys()].every((i) => sentCtx[`tag.t${i}`] === "v"));
  assert.deepEqual(Object.keys(sentCtx).filter((k) => k.startsWith("c")), Object.keys(crowd).slice(0, 10));
  assert.equal(warnings.filter((w) => w.includes("more than the 32")).length, 1);
});

test("export: every outgoing event as a JSON line, in local and cloud mode, after capture and the redact hook", async () => {
  const fs = await import("node:fs");
  const os = await import("node:os");
  const path = await import("node:path");
  const { pathToFileURL } = await import("node:url");
  for (const bad of ["http://example.com/x", "/var/log/fences", "file://otherhost/x"]) {
    assert.throws(() => init({ localOnly: true, export: bad }), /export/, bad);
  }
  const read = (base) => {
    const dir = path.join(base, "exportAgent");
    const files = fs.readdirSync(dir);
    assert.deepEqual(files, [`${new Date().toISOString().slice(0, 10)}.jsonl`]);
    return fs.readFileSync(path.join(dir, files[0]), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  };
  const lookup = tool(function lookup(q) { return q; });
  const exportAgent = governed(async function exportAgent() {
    lookup("cats");
    logDecision("looked it up", "lookup");
    throw new Error("no cats");
  }, { budgetUsd: 1 });
  const run = async () => {
    await context({ user_id: "u_1" }, () => assert.rejects(exportAgent()));
    await flush();
  };

  // Local mode: the same events a server would get, one line each, with the run's id
  let base = fs.mkdtempSync(path.join(os.tmpdir(), "fences-export-"));
  init({ localOnly: true, export: pathToFileURL(base).href, tags: { team: "ops" } });
  await run();
  let lines = read(base);
  assert.deepEqual(lines.map((l) => l.type), ["run_start", "span", "decision", "span", "run_end"]);
  assert.equal(new Set(lines.map((l) => l.run_id)).size, 1);
  assert.ok(lines.every((l) => typeof l.ts === "number"));
  assert.deepEqual(lines[0].context, { user_id: "u_1", "tag.team": "ops" });
  assert.equal(lines[0].budget_usd, 1);
  assert.equal(lines[1].name, "execute_tool lookup");
  assert.equal(lines[2].reasoning, "looked it up");
  assert.equal(lines[3].name, "fences.run.error");
  assert.deepEqual([lines[4].status, lines[4].error], ["error", "Error: no cats"]);

  // Cloud mode: written as well as sent, after capture "metadata" and the redact hook
  base = fs.mkdtempSync(path.join(os.tmpdir(), "fences-export-"));
  cloud({ export: pathToFileURL(base).href, capture: "metadata",
          redact: (e) => (e.type === "span" ? { ...e, attributes: { ...e.attributes, hooked: true } } : e) });
  await run();
  lines = read(base);
  assert.deepEqual(lines.map((l) => l.type), ["run_start", "span", "span", "span", "run_end"], "a decision is a span here");
  for (const secret of ["cats", "looked it up", "u_1"]) assert.ok(!JSON.stringify(lines).includes(secret), secret);
  assert.ok(lines.filter((l) => l.type === "span").every((l) => l.attributes.hooked));
  const strip = ({ type, run_id, ts, ...rest }) => rest;
  assert.deepEqual(lines.filter((l) => l.type === "span").map(strip), sent("/spans").flatMap((b) => b.body.spans), "exactly what was sent");
  assert.equal(lines.at(-1).error, sent("/end")[0].body.error);

  // A write that fails warns once, and the agent carries on
  const blocker = path.join(base, "not_a_dir");
  fs.writeFileSync(blocker, "");
  init({ localOnly: true, export: pathToFileURL(blocker).href });
  warnings.length = 0;
  await run();
  await run();
  assert.equal(warnings.filter((w) => w.includes("couldn't write the trace export")).length, 1);
  init({ localOnly: true });
});

test("export: an agent's name can't climb out of the export directory", async () => {
  const fs = await import("node:fs");
  const os = await import("node:os");
  const path = await import("node:path");
  const { pathToFileURL } = await import("node:url");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fences-export-"));
  init({ localOnly: true, export: pathToFileURL(base).href });
  await governed(async () => 1, { budgetUsd: 1, name: "../../evil" })();
  await governed(async () => 1, { budgetUsd: 1, name: ".." })();
  await flush();
  assert.deepEqual(fs.readdirSync(base).sort(), [".._.._evil", "_"]);
  init({ localOnly: true });
});
