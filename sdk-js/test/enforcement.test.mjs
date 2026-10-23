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
  heartbeatCommands: [], checkpointCommands: [], guardRoute: true, needsApproval: false, down: false, loseAcks: false, breach: null,
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
        if (server.breach) return reply(200, { ok: false, breach: server.breach, commands: out });
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
  assert.equal(check.arguments, "0.4");  // the check gets values (rules need them); the span only names
});

test("mcp.wrap: every callTool is a tool span; named tools are guarded; isError is a failed call", async () => {
  const calls = [];
  const client = fences.mcp.wrap({  // stands in for the MCP SDK's Client
    async callTool({ name }) { calls.push(name); return { isError: name === "broken", content: [{ type: "text", text: "disk full" }] }; },
  }, { guard: ["send_email"], costUsd: { refund: (a) => a.amount } });
  assert.equal((await client.callTool({ name: "search", arguments: { q: "x" } })).isError, false);  // outside a run: straight through

  cloud();
  const [failed, denied, spans] = await governed(async () => {
    await client.callTool({ name: "search", arguments: { q: "cats" } });
    const failed = await client.callTool({ name: "broken" });
    await client.callTool({ name: "send_email", arguments: { to: "a@b.c" } });
    await client.callTool({ name: "constructor" });  // not a guarded tool, though it's a key of every object
    await client.callTool({ name: "refund", arguments: { amount: 0.7 } });
    const denied = await blocked(() => client.callTool({ name: "refund", arguments: { amount: 0.7 } }));
    return [failed, denied, getActiveRun().events];
  }, { budgetUsd: 1 })();
  assert.ok(failed.isError && denied?.breachType === "budget_exceeded");
  assert.deepEqual(calls, ["search", "search", "broken", "send_email", "constructor", "refund"]);  // the refused refund never ran
  assert.deepEqual(spans.map((s) => [s.attributes["gen_ai.tool.name"], s.status]),
    [["search", "ok"], ["broken", "error"], ["send_email", "ok"], ["constructor", "ok"], ["refund", "ok"], ["refund", "error"]]);
  assert.equal(spans[0].attributes["gen_ai.tool.call.arguments"], "q");  // names by default
  assert.equal(spans[1].attributes["error.type"], "MCPToolError");
  assert.deepEqual(server.guards.filter((g) => g.phase === "check").map((g) => [g.tool, g.amount_usd]), [["send_email", 0], ["refund", 0.7]]);
  assert.deepEqual([Math.round(server.spent * 1e9) / 1e9, server.held.size], [0.7, 0]);
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
  cloud();  // a fresh client: the outage opened this one's circuit breaker
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

test("after a halt, the agent's code left running can't call the model, and a tool it aborted is charged", async () => {
  cloud({ stopGraceS: 0.2 }, 100);
  const precheck = await import("../dist/precheck.js");
  const pay = tool(() => new Promise((_, fail) => {  // aborted by the halt, after the payment may have gone out
    getActiveRun().signal.addEventListener("abort", () => fail(new DOMException("aborted", "AbortError")));
  }), { name: "pay", costUsd: 5 });
  let detached;
  const agent = governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.01 });
    const run = getActiveRun();
    detached = (async () => {  // goes on after the governed call is over: JS can't stop it
      await waitFor(() => run.control.cancelled);
      try { precheck.holdNow(precheck.prepare("gpt-4o", "hi", { max_tokens: 10 })); return "sent"; } catch (e) { return e.constructor.name; }
    })();
    return pay();
  }, { budgetUsd: 100 });
  const operator = (async () => {
    await waitFor(() => server.beats.at(-1)?.length);
    server.heartbeatCommands = [{ run_id: server.beats.at(-1)[0].run_id, type: "stop", id: "s2" }];
  })();
  assert.ok((await agent().catch((e) => e)) instanceof FencesStop);
  await operator;
  assert.equal(await detached, "FencesStop");
  await waitFor(() => server.guards.some((g) => g.phase === "settle"));
  const settle = server.guards.find((g) => g.phase === "settle");
  assert.deepEqual([settle.outcome, settle.cost_usd], ["unknown", 5]);
});

test("an MCP policy given as a bare string (untyped JS) names one tool", async () => {
  cloud();
  const called = [];
  const client = fences.mcp.wrap({ callTool: async (p) => { called.push(p.name); return { content: [] }; } }, { guard: "refund" });
  await governed(async () => client.callTool({ name: "refund", arguments: {} }))();
  assert.ok(server.guards.some((g) => g.tool === "refund" && g.phase === "check"), "refund is guarded");
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

// ── waiting at a limit (onBreach: "hold") ─────────────────────────────────────

const heldRun = (kind) => server.beats.flat().findLast((r) => r.held === kind && r.paused)?.run_id;

test("a run held at a breach carries on once the limit is raised and it's resumed", async () => {
  cloud({}, 10);
  const agent = governed(async () => {
    server.breach = "budget_exceeded";
    const held = await checkpoint(null, { costDeltaUsd: 0.01 });  // waits here
    return [held, await checkpoint(null, { costDeltaUsd: 0.01 })];
  }, { budgetUsd: 1, onBreach: "hold" });
  const operator = (async () => {
    await waitFor(() => heldRun("breach"));
    server.breach = null;  // "Raise limits & resume": back to running, then resumed
    server.heartbeatCommands = [{ run_id: heldRun("breach"), type: "resume", id: "r1" }];
  })();
  const [held, after] = await agent();
  await operator;
  assert.ok(held.ok && after.ok, JSON.stringify([held, after]));
  assert.equal(server.ends.at(-1).status, "success");
});

test("a hold ends on a stop, or after pauseTimeoutS with the limit it was held at", async () => {
  cloud({ pauseTimeoutS: 0.3 }, 10);
  const timedOut = governed(async () => {
    server.breach = "iteration_limit";
    const r = await checkpoint(null, { costDeltaUsd: 0.01 });
    server.breach = null;  // raised later: fine again, and not stuck as "paused"
    return [r, await checkpoint(null, { costDeltaUsd: 0.01 })];
  }, { budgetUsd: 1, onBreach: "hold" });
  const [r, later] = await timedOut();
  assert.equal(r.breachType, "iteration_limit");
  assert.ok(later.ok, JSON.stringify(later));

  cloud({}, 10);
  const stopped = governed(async () => {
    server.breach = "budget_exceeded";
    return checkpoint(null, { costDeltaUsd: 0.01 });
  }, { budgetUsd: 1, onBreach: "hold" });
  const operator = (async () => {
    await waitFor(() => heldRun("breach"));
    server.heartbeatCommands = [{ run_id: heldRun("breach"), type: "stop", id: "s1" }];
  })();
  assert.equal((await stopped()).breachType, "stopped_by_user");
  await operator;
});

test("no hold where no one can raise the limit; onBreach is checked", async () => {
  init({ localOnly: true, onBreach: "hold" });
  const local = governed(() => checkpoint(null, { costDeltaUsd: 0.02 }), { budgetUsd: 0.01 });
  assert.equal((await local()).breachType, "budget_exceeded", "local mode never holds");
  cloud({ onBreach: "hold" });
  server.down = true;
  assert.equal((await local()).breachType, "budget_exceeded", "a breach found while the server is unreachable never holds");
  assert.throws(() => governed(() => 1, { onBreach: "wait" }), /onBreach/);
});

test("a reply listing resume before limits raises the limit first", async () => {
  cloud({ heartbeatS: 60 }, 10);
  const { apply } = await import("../dist/control.js");
  const seen = await governed(async () => {
    const run = getActiveRun();
    run.control.paused = true;
    const order = [];
    const desc = Object.getOwnPropertyDescriptor(run.control, "paused");
    Object.defineProperty(run.control, "paused", { configurable: true, get: () => desc.value,
      set: (v) => { if (!v) order.push(run.budgetUsd); desc.value = v; } });
    apply(run, [{ type: "resume", id: "r9" }, { type: "limits", budget_usd: 5 }]);
    return order;
  }, { budgetUsd: 0.01 })();
  assert.deepEqual(seen, [5], "the resume landed before the raised limit");
});

test("a guarded tool denied for budget waits for a raise, then runs, re-checked with the same guard id", async () => {
  cloud({ onBreach: "hold" }, 0.3);
  const ran = [];
  const pay = tool(() => ran.push(1), { costUsd: 0.5, name: "pay" });
  const operator = (async () => {
    await waitFor(() => heldRun("budget"));
    server.budget = 5;
    server.heartbeatCommands = [{ run_id: heldRun("budget"), type: "resume", id: "r2" }];
  })();
  await governed(() => pay(), { budgetUsd: 10 })();
  await operator;
  const checks = server.guards.filter((g) => g.phase === "check");
  assert.deepEqual(ran, [1]);
  assert.ok(checks.length >= 2 && new Set(checks.map((g) => g.guard_id)).size === 1);
});

test("the circuit breaker: after 3 failed calls in a row, calls are skipped until one test call gets through", async () => {
  const { FencesClient, FAILURES_TO_OPEN } = await import("../dist/client.js");
  const client = new FencesClient("fc_test", base);
  let tried = 0;
  const realFetch = globalThis.fetch;
  globalThis.fetch = (...a) => { tried += 1; return realFetch(...a); };
  try {
    server.down = true;  // 503s
    for (let i = 0; i < FAILURES_TO_OPEN; i++) assert.ok((await client.checkpoint("r", {})).network_error);
    assert.equal(tried, 3);
    assert.match((await client.checkpoint("r", {})).network_error, /calls paused/);
    assert.equal(tried, 3, "skipped while open: no waiting on a timeout");
    client.breaker.openUntil = 0;  // the pause is over: one test call, which fails, and it's open again
    await client.checkpoint("r", {});
    await client.checkpoint("r", {});
    assert.equal(tried, 4);
    server.down = false;
    client.breaker.openUntil = 0;  // back up: closed again
    await client.checkpoint("r", {});
    await client.checkpoint("r", {});
    assert.equal(tried, 6);
    assert.equal(client.breaker.failures, 0);
    server.rateLimitAcks = true;  // a heartbeat over its own budget (429) doesn't count against checkpoints
    await client.heartbeat([], [{ id: "x" }]);
    assert.equal(client.breaker.failures, 0);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("heartbeats back off while they fail, and recover once the server answers", async () => {
  cloud({ heartbeatS: 0.05 });
  server.down = true;
  const client = (await import("../dist/core.js"))._client();
  const attempts = [];
  const heartbeat = client.heartbeat.bind(client);
  client.heartbeat = (...a) => { attempts.push(Date.now()); return heartbeat(...a); };  // counted before the breaker
  await governed(async () => {
    await checkpoint(null, { costDeltaUsd: 0.01 });
    await sleep(1500);  // without backoff: about 30 heartbeats
  }, { budgetUsd: 1 })();
  server.down = false;
  assert.ok(attempts.length >= 2 && attempts.length <= 9, String(attempts.length));
  const gaps = attempts.slice(1).map((t, i) => t - attempts[i]);
  assert.ok(gaps.at(-1) > gaps[0], JSON.stringify(gaps));  // the waits grow
});
