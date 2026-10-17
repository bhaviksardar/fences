/**
 * HTTP calls to a Fences server, as specified in PROTOCOL.md, with the fetch built into Node 18+.
 * Nothing throws except a rejected key at run start.
 */
import { FencesAuthError } from "./errors.js";
import { VERSION } from "./version.js";

export type Reply = Record<string, any>;  // eslint-disable-line @typescript-eslint/no-explicit-any

export const FAILURES_TO_OPEN = 3;  // failed calls in a row before the breaker opens
const OPEN_MS = 30_000;             // how long calls are skipped once it's open (with jitter), before one goes through to test

/**
 * A circuit breaker: after FAILURES_TO_OPEN failed calls in a row (no connection, a timeout, a 5xx,
 * or a 429 on anything but the heartbeat), skip calls for about 30 seconds, then let one through to test the server. While it's open
 * a call fails at once instead of after the timeout: checkpoints are enforced locally and guarded
 * tools are blocked, without adding seconds to every step.
 */
export class Breaker {
  failures = 0;
  openUntil = 0;
  private probing = false;

  allow(): boolean {
    if (this.failures < FAILURES_TO_OPEN) return true;
    if (this.probing || Date.now() < this.openUntil) return false;
    this.probing = true;  // one test call at a time
    return true;
  }

  record(ok: boolean): void {
    const wasOpen = this.failures >= FAILURES_TO_OPEN;
    this.probing = false;
    if (ok) {
      this.failures = 0;
      if (wasOpen) console.warn("agentfences: Fences is reachable again");
      return;
    }
    this.failures += 1;
    if (this.failures < FAILURES_TO_OPEN) return;
    this.openUntil = Date.now() + OPEN_MS * (0.75 + Math.random() / 2);  // jitter: SDKs cut off together don't all come back at once
    if (!wasOpen) {
      console.warn(`agentfences: Fences failed ${this.failures} calls in a row; skipping calls for about ${OPEN_MS / 1000}s ` +
        "(limits enforced locally, guarded tools blocked)");
    }
  }
}

export class FencesClient {
  private readonly headers: Record<string, string>;
  readonly breaker = new Breaker();
  /** The server echoed a step_id: it counts a repeated step once, so retrying a checkpoint is safe. */
  dedupesSteps = false;

  constructor(apiKey: string, readonly endpoint: string, private readonly timeoutMs = 3000) {
    this.endpoint = endpoint.replace(/\/+$/, "");
    // The server uses the SDK version to warn about old SDKs and to skip commands they don't understand
    this.headers = { "Content-Type": "application/json", "X-API-Key": apiKey, "X-Fences-SDK": `js/${VERSION}` };
  }

  startRun(body: Record<string, unknown>): Promise<Reply> { return this.post("/api/runs/start", body, false, true); }
  /** body.step_id, if set, lets the server count the step once however often it's sent. */
  async checkpoint(runId: string, body: Record<string, unknown>): Promise<Reply> {
    const resp = await this.post(`/api/runs/${runId}/checkpoint`, body, false, false, !!body.step_id && this.dedupesSteps);
    if (body.step_id && resp.step_id === body.step_id) this.dedupesSteps = true;
    return resp;
  }
  logDecision(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/decisions`, body); }
  endRun(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/end`, body); }
  logSpans(runId: string, spans: unknown[]): Promise<Reply> { return this.post(`/api/runs/${runId}/spans`, { spans }, true); }
  /** Report live runs and acknowledge commands; the reply may carry commands for them. */
  heartbeat(runs: unknown[], acks: unknown[] = []): Promise<Reply> {
    return this.post("/api/heartbeat", acks.length ? { runs, acks } : { runs }, true);
  }
  /** Check a guarded tool call before it runs ({ phase: "check" }), or settle it after ({ phase: "settle" }). */
  guard(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/guard`, body, true, false, true); }  // guard_id: safe to repeat
  requestApproval(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/approvals`, body, true); }

  /**
   * optional: an endpoint older servers may not have. A missing route returns { unsupported: true },
   * and any other error { ok: false, status }.
   * start: a rejected key throws FencesAuthError, before any agent code runs. Later in a run it
   * returns { network_error, auth_rejected: true }: throwing there would replace the agent's own result or error.
   * retry: a call that's safe to repeat goes once more after a connection error (not after a timeout,
   * which would double the wait).
   */
  async post(path: string, payload: unknown, optional = false, start = false, retry = false): Promise<Reply> {
    if (!this.breaker.allow()) return { network_error: "Fences unreachable: calls paused after repeated failures" };
    let json: string;
    try {
      json = JSON.stringify(payload);
    } catch (e) {  // a payload JSON can't encode: not the server's fault
      this.breaker.record(true);
      return { network_error: String(e) };
    }
    const send = () => fetch(this.endpoint + path, {
      method: "POST", headers: this.headers, body: json, signal: AbortSignal.timeout(this.timeoutMs),
    });
    let resp: Response;
    try {
      try {
        resp = await send();
      } catch (e) {
        if (!retry || (e as Error)?.name === "TimeoutError") throw e;
        await new Promise((ok) => setTimeout(ok, 100 + Math.random() * 200));  // a blip: once more, after a moment
        resp = await send();
      }
    } catch (e) {
      this.breaker.record(false);
      return { network_error: String(e) };
    }
    // A 429 on the heartbeat is its own budget, not an outage: it mustn't cut off checkpoints
    this.breaker.record(resp.status < 500 && (resp.status !== 429 || path === "/api/heartbeat"));
    const text = await resp.text().catch(() => "");
    let body: Reply = {}, isJson = false;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) [body, isJson] = [parsed, true];  // a list isn't a Fences reply
    } catch { /* not JSON: the status code still decides */ }
    const detail = typeof body.detail === "string" ? body.detail : "";
    if (optional) {
      if ((resp.status === 404 || resp.status === 405) && (detail === "Not Found" || detail === "Method Not Allowed")) {
        return { unsupported: true };
      }
      return resp.ok ? body : { ok: false, status: resp.status };
    }
    if (resp.status === 401 || resp.status === 403) {
      if (start) throw new FencesAuthError(`Fences API key rejected: ${detail || text.slice(0, 200)}`);
      return { network_error: `API key rejected (HTTP ${resp.status})`, auth_rejected: true };
    }
    if (resp.status === 423) return { ok: false, quarantined: true, detail };  // the agent is quarantined: the run must not start
    if (resp.status === 409) return { ok: false, conflict: detail };          // e.g. checkpoint on a run already fenced
    if (!resp.ok || !isJson) return { network_error: resp.ok ? "response was not JSON" : `HTTP ${resp.status}` };
    return body;
  }
}
