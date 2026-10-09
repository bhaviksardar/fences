/** HTTP calls to a Fences server, as specified in PROTOCOL.md. Uses the fetch built into Node 18+. */
import { FencesAuthError } from "./errors.js";
import { VERSION } from "./version.js";

export type Reply = Record<string, any>;  // eslint-disable-line @typescript-eslint/no-explicit-any

export class FencesClient {
  private readonly headers: Record<string, string>;

  constructor(apiKey: string, readonly endpoint: string, private readonly timeoutMs = 3000) {
    this.endpoint = endpoint.replace(/\/+$/, "");
    // The server uses the SDK version to warn about old SDKs and to skip commands they don't understand
    this.headers = { "Content-Type": "application/json", "X-API-Key": apiKey, "X-Fences-SDK": `js/${VERSION}` };
  }

  startRun(body: Record<string, unknown>): Promise<Reply> { return this.post("/api/runs/start", body); }
  checkpoint(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/checkpoint`, body); }
  logDecision(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/decisions`, body); }
  endRun(runId: string, body: Record<string, unknown>): Promise<Reply> { return this.post(`/api/runs/${runId}/end`, body); }
  logSpans(runId: string, spans: unknown[]): Promise<Reply> { return this.post(`/api/runs/${runId}/spans`, { spans }, true); }

  /**
   * optional: an endpoint older servers may not have. A missing route returns
   * { unsupported: true } and nothing throws, not even a rejected key.
   */
  async post(path: string, payload: unknown, optional = false): Promise<Reply> {
    let resp: Response;
    try {
      resp = await fetch(this.endpoint + path, {
        method: "POST", headers: this.headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (e) {
      return { network_error: String(e) };
    }
    const text = await resp.text().catch(() => "");
    let body: Reply = {}, isJson = false;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed === "object") [body, isJson] = [parsed, true];
    } catch { /* not JSON: the status code still decides */ }
    const detail = typeof body.detail === "string" ? body.detail : "";
    if (optional) {
      if ((resp.status === 404 || resp.status === 405) && (detail === "Not Found" || detail === "Method Not Allowed")) {
        return { unsupported: true };
      }
      return resp.ok ? body : { ok: false, status: resp.status };
    }
    if (resp.status === 401 || resp.status === 403) throw new FencesAuthError(`Fences API key rejected: ${text}`);
    if (resp.status === 423) return { ok: false, quarantined: true, detail };  // the agent is quarantined: the run must not start
    if (resp.status === 409) return { ok: false, conflict: detail };          // e.g. checkpoint on a run already fenced
    if (!resp.ok || !isJson) return { network_error: resp.ok ? "response was not JSON" : `HTTP ${resp.status}` };
    return body;
  }
}
