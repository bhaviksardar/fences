// Live control against a fake Fences server: heartbeat, stop, pause, limits, approvals.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { execFile } from "node:child_process";
import * as fences from "../dist/index.js";

const { governed, checkpoint, init, getActiveRun, requestApproval } = fences;
const warnings = [];
console.warn = (m) => warnings.push(String(m));
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
async function waitFor(cond, ms = 3000) {
  const end = Date.now() + ms;
  while (!cond()) { assert.ok(Date.now() < end, "timed out waiting"); await sleep(10); }
}

const server = {};
const reset = () => Object.assign(server, { beats: [], approvals: [], heartbeatCommands: [], checkpointCommands: [],
                                             heartbeatRoute: true, approvalsRoute: true });
let base, srv;
before(async () => {
  srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => raw += c);
    req.on("end", () => {
      const body = JSON.parse(raw || "{}");
      const reply = (status, obj) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
      if (req.url === "/api/heartbeat") {
        if (!server.heartbeatRoute) return reply(404, { detail: "Not Found" });
        server.beats.push(body.runs);
        const out = server.heartbeatCommands; server.heartbeatCommands = [];
        return reply(200, { commands: out });
      }
      if (req.url.endsWith("/approvals")) {
        if (!server.approvalsRoute) return reply(404, { detail: "Not Found" });
        server.approvals.push({ runId: req.url.split("/")[3], ...body });
        return reply(200, { ok: true });
      }
      if (req.url.endsWith("/checkpoint")) {
        const out = server.checkpointCommands; server.checkpointCommands = [];
        return reply(200, { ok: true, commands: out });
      }
      reply(200, { ok: true });
    });
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${srv.address().port}`;
});
after(() => { init({ localOnly: true }); srv.close(); });
const cloud = (opts = {}) => { reset(); init({ apiKey: "fc_test", endpoint: base, heartbeatS: 0.05, ...opts }); };

test("the heartbeat reports live runs and brings back limits and stop", async () => {
  cloud();
  const agent = governed(async () => {
    for (let step = 0; step < 1000; step++) {
      const r = await checkpoint(null, { costDeltaUsd: 0.001 });
      if (r.breached) return [step, r, getActiveRun().budgetUsd];
      await sleep(20);
    }
  }, { budgetUsd: 1 });
  const operator = (async () => {
    await waitFor(() => server.beats.length);
    const run = server.beats.at(-1)[0];
    assert.deepEqual(Object.keys(run).sort(), ["iterations", "paused", "run_id", "spent_usd", "tokens_used"]);
    server.heartbeatCommands = [{ run_id: run.run_id, type: "limits", budget_usd: 5 }];
    await waitFor(() => !server.heartbeatCommands.length);
    server.heartbeatCommands = [{ run_id: run.run_id, type: "stop" }];
  })();
  const [steps, result, budget] = await agent();
  await operator;
  assert.equal(result.breachType, "stopped_by_user");
  assert.equal(budget, 5);
  assert.ok(steps < 200, "stopped promptly, though every checkpoint reply said ok");
});

test("pause holds the agent at its checkpoint until resumed; a forgotten pause times out", async () => {
  cloud();
  const agent = governed(async () => {
    const t = Date.now();
    server.checkpointCommands = [{ type: "pause" }];  // a checkpoint reply can carry commands too
    const r = await checkpoint(null, { costDeltaUsd: 0.01 });
    return [r, Date.now() - t];
  }, { budgetUsd: 1 });
  const operator = (async () => {
    await waitFor(() => server.beats.at(-1)?.[0]?.paused);  // the heartbeat says it's paused
    await sleep(300);
    server.heartbeatCommands = [{ run_id: server.beats.at(-1)[0].run_id, type: "resume" }];
  })();
  const [result, waited] = await agent();
  await operator;
  assert.ok(result.ok && waited >= 300, `${JSON.stringify(result)} after ${waited}ms`);

  cloud({ pauseTimeoutS: 0.3 });
  const forgotten = await governed(async () => {
    server.checkpointCommands = [{ type: "pause" }];
    return checkpoint(null, { costDeltaUsd: 0.01 });
  }, { budgetUsd: 1 })();
  assert.equal(forgotten.breachType, "paused");
  assert.match(forgotten.message, /not resumed in time/);
});

test("approvals: granted raises the budget; no answer, local mode and old servers are denied", async () => {
  init({ localOnly: true });
  const ask = (timeoutS) => governed(async () => {
    const a = await requestApproval("Refund $240 to order 1182?", { amountUsd: 240, timeoutS });
    return [a, getActiveRun().budgetUsd, getActiveRun().events];
  }, { budgetUsd: 1 })();
  let [a] = await ask();
  assert.ok(!a.granted && /Local mode/.test(a.note));
  assert.ok(!(await requestApproval("outside a run")).granted);

  cloud();
  const approver = (async () => {
    await waitFor(() => server.approvals.length);
    const req = server.approvals.at(-1);
    assert.deepEqual([req.reason, req.amount_usd, req.timeout_s], ["Refund $240 to order 1182?", 240, 600]);
    await sleep(100);
    server.heartbeatCommands = [{ run_id: req.runId, type: "approval", approval_id: req.approval_id, granted: true,
                                  by: "dana@example.com", note: "ok, loyal customer", amount_usd: 240 }];
  })();
  let budget, events;
  [a, budget, events] = await ask(600);
  await approver;
  assert.ok(a.granted && a.by === "dana@example.com");
  assert.equal(budget, 241);
  const span = events.find((e) => e.name === "fences.approval");
  assert.ok(span.attributes["fences.approval.granted"] && span.end_ts - span.start_ts >= 0.1);

  [a] = await ask(0.3);  // nobody answers
  assert.ok(!a.granted && /timeout/.test(a.note));

  server.approvalsRoute = false;
  warnings.length = 0;
  [a] = await ask();
  assert.ok(!a.granted && /doesn't support approvals/.test(a.note));
  assert.equal(warnings.filter((w) => w.includes("doesn't support approval requests yet, so they are denied")).length, 1);
});

test("a server without the heartbeat gets one warning and runs carry on", async () => {
  cloud();
  server.heartbeatRoute = false;
  warnings.length = 0;
  const done = await governed(async () => {
    for (let i = 0; i < 10; i++) { await sleep(30); assert.ok((await checkpoint(null, { costDeltaUsd: 0.01 })).ok); }
    return "done";
  }, { budgetUsd: 1 })();
  assert.equal(done, "done");
  assert.equal(warnings.filter((w) => w.includes("doesn't support the heartbeat yet")).length, 1);
});

test("the heartbeat never keeps the process alive", async () => {
  const script = `
    import { init, governed, checkpoint } from ${JSON.stringify(new URL("../dist/index.js", import.meta.url).href)};
    init({ apiKey: "fc_test", endpoint: ${JSON.stringify(base)} });
    await governed(async () => checkpoint(null, { costDeltaUsd: 0.01 }), { budgetUsd: 1 })();
    console.log("ran");`;
  const t = Date.now();  // async, so this process's fake server can answer the child
  const r = await new Promise((ok) => execFile(process.execPath, ["--input-type=module", "-e", script], { timeout: 10_000 },
    (error, stdout, stderr) => ok({ error, stdout, stderr })));
  assert.equal(r.stdout.trim(), "ran", r.stderr);
  assert.ok(Date.now() - t < 5000, `took ${Date.now() - t}ms to exit`);
});
