import { getActiveRun } from "./core.js";
import { MAX_ARGS, recordToolCall } from "./spans.js";

/** `{"query":"cats","limit":5}`: the arguments as compact JSON (one object argument on its own), capped. */
export function argsSummary(args: unknown[]): string {
  let text: string;
  try {
    text = JSON.stringify(args.length === 1 ? args[0] : args, (_k, v) => typeof v === "bigint" ? String(v) : v) ?? "";
  } catch {
    text = args.map(String).join(", ");  // circular or odd values
  }
  return text.length <= MAX_ARGS ? text : text.slice(0, MAX_ARGS - 1) + "…";
}

/**
 * Record every call of an agent tool on the active run:
 *
 *     const webSearch = tool(async (query: string) => { ... });
 *     const fetchPage = tool(fetchPageImpl, { name: "fetch" });
 *
 * Records the tool's name, a short summary of its arguments, whether it succeeded (and the
 * error if not) and how long it took. Errors still reach the caller; sync tools stay sync.
 * Outside a governed run it just calls the function.
 */
export function tool<A extends unknown[], R>(fn: (...args: A) => R, options: { name?: string } = {}): (...args: A) => R {
  const name = options.name ?? (fn.name || "tool");
  return (...args: A): R => {
    if (!getActiveRun()) return fn(...args);
    const summary = argsSummary(args), started = performance.now();
    let out: R;
    try {
      out = fn(...args);
    } catch (e) {
      recordToolCall(name, summary, started, e);
      throw e;
    }
    if (out instanceof Promise) {
      return out.then((v) => { recordToolCall(name, summary, started); return v; },
                      (e) => { recordToolCall(name, summary, started, e); throw e; }) as R;
    }
    recordToolCall(name, summary, started);
    return out;
  };
}
