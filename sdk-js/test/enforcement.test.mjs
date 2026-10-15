// The TypeScript side of ENFORCEMENT.md: guarded tools, timeouts, stops that reach a stuck run,
// acknowledgements and containment hooks, against a fake Fences server over real HTTP.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { execFile } from "node:child_process";
import * as fences from "../dist/index.js";

const { governed, checkpoint, init, getActiveRun, tool, ToolBlocked, FencesStop } = fences;
const warnings = [];
console.warn = (m) => warnings.push(String(m));
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
async function waitFor(cond, ms = 5000) {
  const end = Date.now() + ms;
  while (!cond()) { assert.ok(Date.now() < end, "timed out waiting"); await sleep(10); }
}

// ── fake server: reserves guarded amounts atomically, hands out commands, records acks ──
const server = {};
const reset = (budget = 1) => Object.assign(server, {
  budget, spent: 0, held: new Map(), guards: [], ends: [], beats: [], acks: [], approvals: [],
  heartbeatCommands: [], checkpointCommands: [], guardRoute: true, needsApproval: false, down: false, loseAcks: false,
});
let base, srv;
before(async () => {
  srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => raw += c);
    req.on("end", () => {
      const body = JSON.parse(raw || "{}");
      const reply = (status, obj) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
      if (server.down) return reply(503, { detail: "down" });
      if (req.url === "/api/heartbeat") {
        if (body.acks && server.loseAcks) { server.loseAcks = false; return reply(503, { detail: "lost" }); }
        if (body.acks && server.rateLimitAcks) { server.rateLimitAcks = false; return reply(429, { detail: "Too many heartbeats" }); }
        server.beats.push(body.runs);
        server.acks.push(...(body.acks ?? []));
        const out = server.heartbeatCommands; server.heartbeatCommands = [];
        return reply(200, { commands: out });
      }
      if (req.url.endsWith("/guard")) {
        if (!server.guardRoute) return reply(404, { detail: "Not Found" });
        server.guards.push(body);
        if (body.phase === "settle") {
          server.held.delete(body.guard_id);
          server.spent += body.cost_usd;
          return reply(200, { ok: true });
        }
        if (server.needsApproval && !body.approval_id) return reply(200, { allow: false, needs_approval: true, reason: "needs approval" });
        const held = [...server.held.values()].reduce((a, b) => a + b, 0);
        if (server.spent + held + body.amount_usd > server.budget) return reply(200, { allow: false, reason: "over budget", breach: "budget_exceeded" });
        server.held.set(body.guard_id, body.amount_usd);
        return reply(200, { allow: true });
      }
      if (req.url.endsWith("/approvals")) {
        server.approvals.push({ runId: req.url.split("/")[3], ...body });
        return reply(200, { ok: true });
      }
      if (req.url.endsWith("/checkpoint")) {
        const out = server.checkpointCommands; server.checkpointCommands = [];
        return reply(200, { ok: true, commands: out });
      }
      if (req.url.endsWith("/end")) server.ends.push(body);
      reply(200, { ok: true });
    });
  });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${srv.address().port}`;
});
after(() => { init({ localOnly: true }); srv.close(); });
const cloud = (opts = {}, budget = 1) => { reset(budget); init({ apiKey: "fc_test", endpoint: base, heartbeatS: 0.05, ...opts }); };
const blocked = async (fn) => { try { await fn(); return null; } catch (e) { if (e instanceof ToolBlocked) return e; throw e; } };

// ── guarded tools ─────────────────────────────────────────────────────────────

test("parallel guarded calls can't overspend together, locally or against a server", async () => {
  const ran = [];
  const refund = tool(async (amount) => { await sleep(50); ran.push(amount); }, { costUsd: (amount) => amount, name: "refund" });
  const agent = governed(async () => {
    const results = await Promise.allSettled(Array.from({ length: 5 }, () => refund(0.4)));
    const run = getActiveRun();
    return [results.filter((r) => r.status === "rejected" && r.reason instanceof ToolBlocked).length, run.costUsd, run.reservedUsd];
  }, { budgetUsd: 1 });

  init({ localOnly: true });
  let [refused, spent, held] = await agent();
  assert.deepEqual([ran.length, refused, Math.round(spent * 1e9) / 1e9, held], [2, 3, 0.8, 0]);

  cloud();
  ran.length = 0;
  [refused] = await agent();
  assert.deepEqual([ran.length, refused, Math.round(server.spent * 1e9) / 1e9, server.held.size], [2, 3, 0.8, 0]);
  const check = server.guards.find((g) => g.phase === "check");
  assert.deepEqual(Object.keys(check).sort(), ["amount_usd", "approval_id", "arguments", "guard_id", "phase", "tool"]);
  assert.equal(check.tool, "refund");
});

test("a blocked tool never runs; outside a run it just runs; local approval is denied", async () => {
  init({ localOnly: true });
  const ran = [];
  const deleteRecords = tool(() => ran.push("delete"), { guard: true });
  const overBudget = governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.1 });
    return blocked(deleteRecords);
  }, { budgetUsd: 0.05 });
  const e = await overBudget();
  assert.ok(e && e.breachType === "budget_exceeded" && !ran.length);
  await deleteRecords();
  assert.deepEqual(ran, ["delete"], "outside a governed run a guarded tool just runs");

  const wire = tool(() => ran.push("wire"), { approve: true });
  assert.match((await governed(() => blocked(wire), { budgetUsd: 1 })()).reason, /Local mode/);
});

test("a failed call is released uncharged; a timed-out one is charged; timeouts reject in time", async () => {
  init({ localOnly: true });
  const declined = tool(async () => { throw new Error("card declined"); }, { costUsd: 0.3 });
  const slowPayment = tool(() => new Promise((ok) => setTimeout(ok, 5000).unref()), { costUsd: 0.3, timeoutS: 0.05 });
  const agent = governed(async () => {
    const run = getActiveRun();
    await declined().catch(() => {});
    assert.deepEqual([run.costUsd, run.reservedUsd], [0, 0]);
    const t = Date.now();
    const err = await slowPayment().catch((e) => e);
    assert.equal(err.name, "TimeoutError");
    assert.ok(Date.now() - t < 1000);
    return [run.costUsd, run.reservedUsd];
  }, { budgetUsd: 1 });
  assert.deepEqual(await agent(), [0.3, 0]);
});

test("guards fail closed: unreachable or unsupported servers block the tool", async () => {
  const ran = [];
  const charge = tool(() => ran.push(1), { costUsd: 0.1, name: "charge_card" });
  const agent = governed(() => blocked(charge), { budgetUsd: 1 });
  cloud();
  server.down = true;
  assert.match((await agent()).reason, /couldn't reach Fences/);
  server.down = false;
  server.guardRoute = false;
  warnings.length = 0;
  assert.match((await agent()).reason, /doesn't support guarded tools/);
  assert.match((await agent()).reason, /doesn't support guarded tools/);
  assert.equal(warnings.filter((w) => w.includes("guarded tools")).length, 1);
  assert.equal(ran.length, 0);
});

test("a tool the server says needs approval: asked, approved, then run once", async () => {
  cloud();
  server.needsApproval = true;
  const ran = [];
  const issueRefund = tool((orderId, amount) => ran.push(orderId), { costUsd: (orderId, amount) => amount, name: "issue_refund" });
  const approver = (async () => {
    await waitFor(() => server.approvals.length);
    const req = server.approvals.at(-1);
    assert.equal(req.reason, 'issue_refund(["1182",0.5])');
    server.heartbeatCommands = [{ run_id: req.runId, type: "approval", approval_id: req.approval_id, granted: true, by: "dana@example.com" }];
  })();
  await governed(() => issueRefund("1182", 0.5), { budgetUsd: 1 })();
  await approver;
  const checks = server.guards.filter((g) => g.phase === "check");
  assert.deepEqual(ran, ["1182"]);
  assert.equal(checks.length, 2);
  assert.equal(checks[1].approval_id, server.approvals.at(-1).approval_id);
});

// ── stops that reach a stuck run ──────────────────────────────────────────────

test("a stop during a hanging tool halts the call with FencesStop and aborts run.signal", async () => {
  cloud({ stopGraceS: 0.2 });
  let signal;
  // A real tool passes the run's signal to its fetches; this one waits on it the same way
  const crawl = tool(() => new Promise((ok, fail) => {
    const t = setTimeout(ok, 60_000, "late");
    getActiveRun().signal.addEventListener("abort", () => { clearTimeout(t); fail(new Error("aborted")); });
  }), { name: "crawl" });
  const agent = governed(async () => {
    signal = getActiveRun().signal;
    await checkpoint(null, { costDeltaUsd: 0.01 });
    return crawl();
  }, { budgetUsd: 1 });
  const operator = (async () => {
    await waitFor(() => server.beats.at(-1)?.length);
    server.heartbeatCommands = [{ run_id: server.beats.at(-1)[0].run_id, type: "stop", id: "s1" }];
  })();
  const t = Date.now();
  const err = await agent().catch((e) => e);
  await operator;
  assert.ok(err instanceof FencesStop && err.breachType === "stopped_by_user", String(err));
  assert.ok(Date.now() - t < 3000);
  assert.ok(signal.aborted && signal.reason instanceof FencesStop);
  assert.deepEqual([server.ends.at(-1).status, server.ends.at(-1).breach], ["breached", "stopped_by_user"]);
  assert.ok(server.beats.some((b) => b.some((r) => r.stopped)), "the heartbeat says the stop was received");
});

test("an agent that saw its stop at a checkpoint gets to wrap up", async () => {
  cloud({ stopGraceS: 0.1 });
  const agent = governed(async () => {
    for (let i = 0; i < 500; i++) {
      const r = await checkpoint(null, { costDeltaUsd: 0.001 });
      if (r.breached) { await sleep(500); return r.breachType; }  // the closing summary
      await sleep(10);
    }
  }, { budgetUsd: 1 });
  const operator = (async () => {
    await waitFor(() => server.beats.at(-1)?.length);
    server.heartbeatCommands = [{ run_id: server.beats.at(-1)[0].run_id, type: "stop" }];
  })();
  assert.equal(await agent(), "stopped_by_user");
  await operator;
});

// ── acknowledgements and containment hooks ────────────────────────────────────

test("commands are acknowledged once; hooks run once and report back; a lost heartbeat's acks are resent", async () => {
  const hooked = [];
  cloud({ onQuarantine: (e) => { hooked.push(e); }, onStop: (e) => { hooked.push(e); throw new Error("vault unreachable"); } }, 10);
  server.loseAcks = true;
  const agent = governed(async () => {
    for (let i = 0; i < 2000; i++) {
      const r = await checkpoint(null, { costDeltaUsd: 0.0001 });
      if (r.breached) return r.breachType;
      await sleep(10);
    }
  }, { budgetUsd: 10 });
  const operator = (async () => {
    await waitFor(() => server.beats.at(-1)?.length === 2);
    const ids = server.beats.at(-1).map((r) => r.run_id);
    const quarantine = ids.map((id) => ({ run_id: id, type: "quarantine", id: "q1", by: "dana@example.com", reason: "leaking data" }));
    server.heartbeatCommands = quarantine;
    await waitFor(() => server.beats.at(-1).every((r) => r.paused));
    server.heartbeatCommands = quarantine;  // re-sent with the same id: not acknowledged again
    await waitFor(() => !server.heartbeatCommands.length);
    server.heartbeatCommands = ids.map((id) => ({ run_id: id, type: "resume", id: `r-${id}` }));
    await waitFor(() => server.beats.at(-1).every((r) => !r.paused));
    server.heartbeatCommands = [{ run_id: ids[0], type: "stop", id: "s1" }, { run_id: ids[1], type: "stop" }];
    return ids;
  })();
  assert.deepEqual(await Promise.all([agent(), agent()]), ["stopped_by_user", "stopped_by_user"]);
  const ids = await operator;
  await waitFor(() => server.acks.some((a) => a.hook === "failed"));
  assert.equal(hooked.filter((e) => e.type === "quarantine").length, 1, "once per process per quarantine");
  assert.equal(hooked.find((e) => e.type === "quarantine").reason, "leaking data");
  assert.equal(hooked.filter((e) => e.type === "stop").length, 2, "onStop runs once per run, with or without an id");
  const receipts = server.acks.filter((a) => !a.hook).map((a) => `${a.id}`).sort();
  assert.deepEqual(receipts, ["q1", "q1", ...ids.map((id) => `r-${id}`), "s1"].sort());
  const hooks = Object.fromEntries(server.acks.filter((a) => a.hook).map((a) => [a.id, a]));
  assert.equal(hooks.q1.hook, "ok");
  assert.equal(hooks.s1.hook, "failed");
  assert.match(hooks.s1.detail, /vault unreachable/);
});

test("a heartbeat rate limited with 429 doesn't lose its acks", async () => {
  cloud();
  server.rateLimitAcks = true;
  server.checkpointCommands = [{ type: "stop", id: "s7" }];
  await governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.01 });
    await waitFor(() => server.acks.length);
  }, { budgetUsd: 1 })();
  assert.equal(server.rateLimitAcks, false, "the first heartbeat with acks was rate limited");
  assert.deepEqual(server.acks.map((a) => a.id), ["s7"]);
});

test("a quarantine re-sent on every reply doesn't make heartbeats back-to-back", async () => {
  cloud({ heartbeatS: 0.5, pauseTimeoutS: 1.5 });
  const agent = governed(async () => {
    await sleep(600);
    const before = server.beats.length, t = Date.now();
    await checkpoint(null, { costDeltaUsd: 0.01 });  // held by the quarantine until pauseTimeoutS
    return [server.beats.length - before, (Date.now() - t) / 1000];
  }, { budgetUsd: 1 });
  const resend = setInterval(() => {  // like the server: quarantine in every reply while it lasts
    const runs = server.beats.at(-1) ?? [];
    server.heartbeatCommands = runs.map((r) => ({ run_id: r.run_id, type: "quarantine", id: "q3" }));
  }, 5);
  const [beats, held] = await agent();
  clearInterval(resend);
  assert.ok(beats <= held / 0.5 + 3, `${beats} heartbeats in ${held}s`);
});

test("a quarantine blocks guarded tools at once, without waiting out its pause", async () => {
  cloud({ heartbeatS: 60 });
  const deploy = tool(() => { throw new Error("ran while quarantined"); }, { guard: true });
  const agent = governed(async () => {
    server.checkpointCommands = [{ type: "quarantine", id: "q2" }];
    const r = checkpoint(null, { costDeltaUsd: 0.01 });  // the quarantine arrives, and pauses this checkpoint
    await waitFor(() => getActiveRun().control.quarantined);
    const e = await blocked(deploy);
    getActiveRun().control.paused = false;  // let the paused checkpoint return
    await r;
    return e;
  }, { budgetUsd: 1 });
  assert.match((await agent()).reason, /quarantined/);
});

test("a script that ends right after a stop finishes its hook and acknowledges it first", async () => {
  reset();
  const script = `
    import { init, governed, checkpoint } from ${JSON.stringify(new URL("../dist/index.js", import.meta.url).href)};
    init({ apiKey: "fc_test", endpoint: ${JSON.stringify(base)}, heartbeatS: 60,
           onStop: async () => { await new Promise((ok) => setTimeout(ok, 300)); console.log("revoked"); } });
    const r = await governed(async () => checkpoint(null, { costDeltaUsd: 0.01 }), { budgetUsd: 1 })();
    console.log(r.breachType);`;
  server.checkpointCommands = [{ type: "stop", id: "s9" }];
  const r = await new Promise((ok) => execFile(process.execPath, ["--input-type=module", "-e", script], { timeout: 20_000 },
    (error, stdout, stderr) => ok({ error, stdout, stderr })));
  assert.deepEqual(r.stdout.trim().split("\n"), ["stopped_by_user", "revoked"], r.stderr);
  await waitFor(() => server.acks.some((a) => a.id === "s9" && a.hook === "ok"));
});
