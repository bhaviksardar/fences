/**
 * Record every OpenAI and Anthropic model call made inside a governed run, with no other code
 * changes: model, tokens, cost, latency, and the error if it failed. Calls don't count as
 * checkpoints and don't add to the run's spend (checkpoint(response) does), but in a run with a
 * budget each one is prechecked: refused with FencesStop if its worst case doesn't fit.
 */
import * as precheck from "./precheck.js";
import { getActiveRun, store, type Hold, type Run } from "./run.js";
import { recordLlmCall } from "./spans.js";

const PATCHED = Symbol.for("agentfences.instrumented");
type Fn = (this: unknown, ...args: unknown[]) => unknown;
type Any = any;  // eslint-disable-line @typescript-eslint/no-explicit-any

/**
 * Patch the SDK behind this client so every model call inside a governed run is recorded and,
 * with a budget, prechecked:
 *
 *     import OpenAI from "openai";
 *     const openai = instrument(new OpenAI());
 *
 * Takes an OpenAI client (chat.completions and responses, including OpenAI-compatible clients)
 * or an Anthropic one (messages). One call covers every client of that SDK; calling it again
 * does nothing. Returns the client.
 */
export function instrument<C extends object>(client: C): C {
  const c = client as Any;
  const provider = c.chat ? "openai" : "anthropic";
  const resources = [c.chat?.completions, c.responses, c.messages].filter((r) => typeof r?.create === "function");
  if (!resources.length) throw new TypeError("instrument() takes an OpenAI or Anthropic client");
  for (const r of resources) {
    const proto = Object.getPrototypeOf(r);
    if (typeof proto?.create === "function" && !proto.create[PATCHED]) proto.create = wrap(proto.create, provider);
  }
  return client;
}

function wrap(create: Fn, provider: string): Fn {
  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    const run = getActiveRun();
    if (!run) return create.apply(this, args);
    const body = (args[0] ?? {}) as Record<string, unknown>;
    const prepared = precheck.prepare(body.model, body, body);
    if (!prepared) return send(this, create, args, provider, run, null);
    let hold: Hold | "wait";
    try {
      hold = precheck.holdNow(prepared);  // FencesStop if the call might not fit the budget
    } catch (e) {
      return refused(e);
    }
    if (hold !== "wait") return send(this, create, args, provider, run, hold);
    // onBreach "hold": wait for a raised budget, then send. Boxed, so nothing awaits the SDK's promise before the caller does
    return later(precheck.holdAsync(prepared).then((h) => ({ out: send(this, create, args, provider, run, h) as Any })));
  };
  (wrapped as Any)[PATCHED] = true;
  return wrapped;
}

/** A call sent once a hold is released: awaitable, with the SDK promise's methods, each forwarded once it's sent. */
function later(sent: Promise<{ out: Any }>): Any {
  const value = () => sent.then(({ out }) => out);
  return {
    then: (ok?: (v: unknown) => unknown, ko?: (e: unknown) => unknown) => value().then(ok, ko),
    catch: (ko?: (e: unknown) => unknown) => value().catch(ko),
    finally: (fn?: () => void) => value().finally(fn),
    withResponse: () => sent.then(({ out }) => out.withResponse()),
    asResponse: () => sent.then(({ out }) => out.asResponse()),
  };
}

/** A refused call rejects like the SDK's own promise would, never throwing synchronously, and keeps its methods. */
function refused(error: unknown): Promise<never> {
  const p = Promise.reject(error) as Any;
  p.withResponse = p.asResponse = () => p;
  return p;
}

/**
 * Make the call and return the SDK's own promise (an APIPromise, with .withResponse()) unchanged,
 * except that awaiting it, or calling withResponse() or asResponse(), also records the call and
 * settles its hold. Nothing reads the response before the caller does: after asResponse(), whose
 * body the caller reads, the call is recorded without usage and its worst case stays held until
 * the next checkpoint.
 */
function send(self: unknown, create: Fn, args: unknown[], provider: string, run: Run, hold: Hold | null): unknown {
  const body = (args[0] ?? {}) as Record<string, unknown>;
  const started = performance.now();
  let done = false;
  const record = (response?: unknown, error?: unknown) => {
    if (done) return;
    done = true;
    store.run(run, () => {
      if (hold) precheck.finish(run, hold, response, error !== undefined);
      recordLlmCall(provider, body.model as string | undefined, started, response, error, { stream: !!body.stream });
    });
  };
  let out: Any;
  try {
    out = create.apply(self, args);
  } catch (e) {
    record(undefined, e);
    throw e;
  }
  if (!out || typeof out.then !== "function") {
    record(out);
    return out;
  }
  const then = out.then.bind(out);
  out.then = (ok?: (v: unknown) => unknown, ko?: (e: unknown) => unknown) => then(
    (v: unknown) => { record(v); return ok ? ok(v) : v; },
    (e: unknown) => { record(undefined, e); if (ko) return ko(e); throw e; },
  );
  // APIPromise has its own catch and finally, which skip then: route them through the one above
  out.catch = (ko?: (e: unknown) => unknown) => Promise.prototype.catch.call(out, ko);
  out.finally = (fn?: () => void) => Promise.prototype.finally.call(out, fn);
  if (typeof out.asResponse === "function") {  // the caller reads the body: the call is recorded without usage
    const asResponse = out.asResponse.bind(out);
    out.asResponse = () => asResponse().then(
      (r: unknown) => { record(); return r; },
      (e: unknown) => { record(undefined, e); throw e; },
    );
  }
  if (typeof out.withResponse === "function") {
    const withResponse = out.withResponse.bind(out);
    out.withResponse = () => withResponse().then(
      (r: Any) => { record(r?.data); return r; },
      (e: unknown) => { record(undefined, e); throw e; },
    );
  }
  return out;
}
