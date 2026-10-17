/**
 * Decisions and spans go to the server in the background, in order, so logDecision() and
 * recorded calls never wait on a network round trip. Consecutive spans of a run go together.
 */
import type { FencesClient } from "./client.js";
import type { Span } from "./run.js";

const SPAN_BATCH = 50;
type Item = { kind: "decision"; client: FencesClient; runId: string; body: Record<string, unknown> }
  | { kind: "span"; client: FencesClient; runId: string; span: Span };
const queue: Item[] = [];
let draining: Promise<void> | null = null;
/** Off once a server says it has no spans endpoint. */
export let spansSupported = true;

export function enqueue(item: Item): void {
  queue.push(item);
  draining ??= drain().finally(() => { draining = null; });
}

/** Wait up to timeoutMs for everything queued to be sent. */
export async function wait(timeoutMs: number): Promise<void> {
  if (draining) await Promise.race([draining, new Promise((r) => setTimeout(r, timeoutMs).unref())]);
}

async function drain(): Promise<void> {
  while (queue.length) {
    const item = queue.shift()!;
    try {
      if (item.kind === "decision") {
        await item.client.logDecision(item.runId, item.body);
        continue;
      }
      const batch = [item.span];
      while (queue[0]?.kind === "span" && queue[0].runId === item.runId && batch.length < SPAN_BATCH) {
        batch.push((queue.shift() as Extract<Item, { kind: "span" }>).span);
      }
      if (spansSupported && (await item.client.logSpans(item.runId, batch)).unsupported) {
        spansSupported = false;
        console.warn("agentfences: this Fences server doesn't accept tool and model call spans yet; " +
          "they're kept on the run locally but not sent");
      }
    } catch { /* never let delivery break the agent */ }
  }
}
