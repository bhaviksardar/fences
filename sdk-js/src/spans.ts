/**
 * Attributes for model and tool call spans, named as in the OpenTelemetry GenAI semantic
 * conventions (gen_ai.*), plus Fences' own (fences.*). Same format as the Python SDK.
 */
import { recordSpan } from "./core.js";
import { costOf, readUsage } from "./pricing.js";

export const MAX_ARGS = 300;  // characters of a tool call's argument summary

/** A `chat {model}` span. started is a performance.now() reading; response carries the usage. */
export function recordLlmCall(provider: string, requestModel: string | undefined, started: number,
                              response?: unknown, error?: unknown, opts: { stream?: boolean; integration?: string } = {}): void {
  const attrs: Record<string, unknown> = { "gen_ai.operation.name": "chat", "gen_ai.provider.name": provider,
    "gen_ai.request.model": requestModel, "fences.integration": opts.integration };
  if (opts.stream) attrs["gen_ai.request.stream"] = true;  // usage arrives with the stream, so none is recorded here
  const usage = response !== undefined && error === undefined ? readUsage(response) : null;
  if (usage) {
    Object.assign(attrs, {
      "gen_ai.response.model": usage.model || undefined,
      "gen_ai.usage.input_tokens": usage.input + usage.cacheRead + usage.cacheWrite,  // OTel counts cached input in the total
      "gen_ai.usage.cache_read.input_tokens": usage.cacheRead,
      "gen_ai.usage.cache_write.input_tokens": usage.cacheWrite,
      "gen_ai.usage.output_tokens": usage.output,
      "fences.cost_usd": costOf(response)[0],
    });
  }
  const model = (attrs["gen_ai.response.model"] as string | undefined) || requestModel || "unknown";
  recordSpan(`chat ${model}`, performance.now() - started, attrs, error);
}

/** An `execute_tool {name}` span; args is a short summary of the call's arguments. */
export function recordToolCall(name: string, args: string, started: number, error?: unknown,
                               opts: { callId?: string; integration?: string } = {}): void {
  recordSpan(`execute_tool ${name}`, performance.now() - started, {
    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.call.id": opts.callId,
    "gen_ai.tool.call.arguments": args.length <= MAX_ARGS ? args : args.slice(0, MAX_ARGS - 1) + "…",
    "fences.integration": opts.integration,
  }, error);
}
