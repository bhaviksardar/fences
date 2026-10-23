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
