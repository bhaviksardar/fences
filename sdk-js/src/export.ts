/**
 * init({ export: "file:///path/to/dir" }): every event that leaves the process (after capture "metadata"
 * and the redact hook) is also written, as one JSON line, to {dir}/{agent}/{YYYY-MM-DD}.jsonl (the UTC
 * date), in local and cloud mode. Written in the background with Node's fs, so the agent never waits on
 * the disk; a failed write warns once and never reaches the agent. Outside Node it does nothing, with a warning.
 */
import { warnOnce } from "./events.js";

const MAX_QUEUED = 10_000;  // a stuck disk can't make a runaway agent eat memory; extra lines are dropped
let dir: URL | null = null;
let queue: { dir: URL; agent: string; line: Record<string, unknown> }[] = [];
let writing: Promise<void> | null = null;
let failed = false;

export function configure(url?: string | null): void {
  dir = null;
  failed = false;
  if (url === undefined || url === null) return;
  let parsed: URL | null = null;
  try { parsed = new URL(url); } catch { /* checked below */ }
  if (!parsed || parsed.protocol !== "file:" || (parsed.host && parsed.host !== "localhost")) {
    throw new TypeError(`export must be a file:// URL of a directory, e.g. file:///var/log/fences, not ${JSON.stringify(url)}`);
  }
  if (typeof process === "undefined" || !process.versions?.node) {
    warnOnce("export-no-fs", "export needs Node's file system; nothing is written here");
    return;
  }
  dir = parsed;
}

export const active = (): boolean => dir !== null;

/** Queue one event, as sent, for the run's file. Never throws. */
export function write(agent: string, runId: string, event: Record<string, unknown>): void {
  if (!dir || queue.length >= MAX_QUEUED) return;
  queue.push({ dir, agent, line: { type: event.type, run_id: runId, ts: Date.now() / 1000, ...event } });
  writing ??= drain().finally(() => { writing = null; });
}

/** Wait up to timeoutMs for everything queued to be written. */
export async function wait(timeoutMs: number): Promise<void> {
  if (writing) await Promise.race([writing, new Promise((r) => setTimeout(r, timeoutMs).unref())]);
}

async function drain(): Promise<void> {
  const [fs, path, url] = await Promise.all([import("node:fs/promises"), import("node:path"), import("node:url")]);
  while (queue.length) {
    const batch = queue;
    queue = [];
    try {
      const files = new Map<string, string[]>();
      for (const { dir, agent, line } of batch) {
        let name = agent.replace(/[^A-Za-z0-9_.-]/g, "_");  // a name can't climb out of the directory
        if (name === "" || name === "." || name === "..") name = "_";
        const day = new Date((line.ts as number) * 1000).toISOString().slice(0, 10);
        const file = path.join(url.fileURLToPath(dir), name, `${day}.jsonl`);
        if (!files.has(file)) files.set(file, []);
        files.get(file)!.push(JSON.stringify(line));
      }
      for (const [file, lines] of files) {
        await fs.mkdir(path.dirname(file), { recursive: true });
        await fs.appendFile(file, lines.join("\n") + "\n", "utf8");
      }
    } catch (e) {  // a full disk or a read-only directory must never break the agent
      if (!failed) {
        failed = true;
        console.warn(`agentfences: couldn't write the trace export (${e instanceof Error ? e.message : String(e)}); events are still sent as usual`);
      }
    }
  }
}
