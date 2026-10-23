/**
 * Turns a model response into [cost in USD, tokens]. Prices come from the same table as the
 * Python SDK (sdk/agentfences/prices.json, a snapshot of LiteLLM's MIT-licensed price list).
 */
import { PRICES, type PriceEntry, type Rates } from "./prices.generated.js";
import { warnOnce } from "./events.js";

/** Per-1M-token prices for models the table doesn't know, from init({ prices }). */
export interface CustomPrice { input: number; output: number; cacheRead?: number; cacheWrite?: number }

const custom: Record<string, PriceEntry> = {};

const validPrice = (v: unknown) => typeof v === "number" && Number.isFinite(v) && v >= 0;

/** input and output are required; a missing or invalid price throws here, not at the agent's first step. */
export function setCustomPrices(prices: Record<string, CustomPrice> = {}): void {
  const table: Record<string, PriceEntry> = {};
  for (const [model, p] of Object.entries(prices)) {
    if (!p || typeof p !== "object" || p.input === undefined || p.output === undefined) {
      throw new TypeError(`init({ prices }): '${model}' needs an input and an output price (USD per 1M tokens)`);
    }
    for (const key of ["input", "output", "cacheRead", "cacheWrite"] as const) {
      if (p[key] !== undefined && !validPrice(p[key])) {
        throw new TypeError(`init({ prices }): '${model}' ${key} must be a number, zero or more, not ${String(p[key])}`);
      }
    }
    const e: PriceEntry = { in: p.input / 1e6, out: p.output / 1e6 };
    if (p.cacheRead !== undefined) e.cache_read = p.cacheRead / 1e6;
    if (p.cacheWrite !== undefined) e.cache_write = p.cacheWrite / 1e6;
    table[model.toLowerCase()] = e;
  }
  for (const k of Object.keys(custom)) delete custom[k];  // only once every entry is valid
  Object.assign(custom, table);
}

export function priceOf(model: string): PriceEntry | undefined {
  const name = model.toLowerCase().split("/").pop() ?? "";            // "models/gemini-2.5-pro", "openai/gpt-4o"
  const undated = name.replace(/-(\d{4}-\d{2}-\d{2}|\d{8})$/, "");     // "claude-sonnet-4-5-20250929"
  for (const candidate of [name, undated]) {
    // hasOwn: a model named "constructor" mustn't find Object.prototype's
    const hit = Object.hasOwn(custom, candidate) ? custom[candidate] : Object.hasOwn(PRICES, candidate) ? PRICES[candidate] : undefined;
    if (hit) return hit;
  }
  return undefined;
}

export interface Usage {
  model: string;
  input: number;        // uncached input
  cacheRead: number;
  cacheWrite: number;
  output: number;
  reportedCost: number | null;  // a gateway's billed cost, when it reports one
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;
const num = (...values: unknown[]): number => {
  for (const v of values) if (typeof v === "number" && Number.isFinite(v)) return v;
  return 0;
};

/**
 * Normalise usage from the common response shapes: OpenAI (Chat Completions, Responses),
 * Anthropic, Gemini, the Vercel AI SDK and LangChain.js, or any object with the same fields.
 * Returns null if the response carries no usage.
 */
export function readUsage(response: unknown): Usage | null {
  const r = response as Any;
  if (!r || typeof r !== "object") return null;
  const model: string = r.model ?? r.modelVersion ?? r.model_version ?? r.response?.modelId
    ?? r.response_metadata?.model_name ?? r.response_metadata?.model ?? "";
  const u = r.usage;

  if (u && typeof u === "object") {
    // Gateways like OpenRouter report the billed cost; only a finite one of zero or more: a NaN or negative one would switch the budget off
    const reported = typeof u.cost === "number" && Number.isFinite(u.cost) && u.cost >= 0 ? u.cost : null;
    if (u.inputTokens !== undefined || u.outputTokens !== undefined) {   // Vercel AI SDK: input includes cached
      const read = num(u.cachedInputTokens, u.inputTokenDetails?.cacheReadTokens);
      const write = num(r.providerMetadata?.anthropic?.cacheCreationInputTokens, u.inputTokenDetails?.cacheWriteTokens);
      return { model, input: num(u.inputTokens) - read - write, cacheRead: read, cacheWrite: write,
               output: num(u.outputTokens), reportedCost: reported };
    }
    const openai = u.prompt_tokens !== undefined ? [u.prompt_tokens_details, u.prompt_tokens, u.completion_tokens]  // Chat Completions
      : u.input_tokens_details !== undefined ? [u.input_tokens_details, u.input_tokens, u.output_tokens] : null;     // Responses API
    if (openai) {  // OpenAI: the input total includes cache reads and writes
      const [details, total, out] = openai;
      const read = num(details?.cached_tokens), write = num(details?.cache_write_tokens);
      return { model, input: num(total) - read - write, cacheRead: read, cacheWrite: write, output: num(out), reportedCost: reported };
    }
    return { model, input: num(u.input_tokens), cacheRead: num(u.cache_read_input_tokens),   // Anthropic: input excludes cache
             cacheWrite: num(u.cache_creation_input_tokens), output: num(u.output_tokens), reportedCost: reported };
  }

  const g = r.usageMetadata ?? r.usage_metadata;
  if (!g || typeof g !== "object") return null;
  if (g.promptTokenCount !== undefined || g.prompt_token_count !== undefined) {  // Gemini: thinking billed as output
    const read = num(g.cachedContentTokenCount, g.cached_content_token_count);
    return { model, input: num(g.promptTokenCount, g.prompt_token_count) - read, cacheRead: read, cacheWrite: 0,
             output: num(g.candidatesTokenCount, g.candidates_token_count) + num(g.thoughtsTokenCount, g.thoughts_token_count),
             reportedCost: null };
  }
  const read = num(g.input_token_details?.cache_read), write = num(g.input_token_details?.cache_creation);  // LangChain.js
  return { model, input: num(g.input_tokens) - read - write, cacheRead: read, cacheWrite: write, output: num(g.output_tokens),
           reportedCost: null };
}

/** [cost in USD, tokens] for one model response. Warns once for a model it can't price. */
export function costOf(response: unknown): [number, number] {
  const u = readUsage(response);
  if (!u) {
    const kind = (response as Any)?.constructor?.name ?? typeof response;
    warnOnce(`no-usage:${kind}`, `${kind} has no token usage, so this step costs $0. For streams, pass the final ` +
      "message or a chunk that includes usage.");
    return [0, 0];
  }
  for (const k of ["input", "cacheRead", "cacheWrite", "output"] as const) u[k] = Math.max(u[k], 0);  // odd counts never lower the cost
  const tokens = u.input + u.cacheRead + u.cacheWrite + u.output;
  if (u.reportedCost !== null) return [u.reportedCost, tokens];
  const p = u.model ? priceOf(u.model) : undefined;
  if (!p) {
    warnOnce(`no-price:${u.model}`, `no price for model '${u.model}'. Its tokens are counted but its cost isn't, so ` +
      `budgetUsd can't stop it. Add it with init({ prices: { "${u.model || "model-name"}": { input: ..., output: ... } } }) ` +
      "(USD per 1M tokens).");
    return [0, tokens];
  }
  const prompt = u.input + u.cacheRead + u.cacheWrite;
  const rates: Rates = p.above !== undefined && p.tier && prompt > p.above ? { ...p, ...p.tier } : p;  // long-context pricing
  const cost = u.input * rates.in + u.cacheRead * (rates.cache_read ?? rates.in)
    + u.cacheWrite * (rates.cache_write ?? rates.in) + u.output * rates.out;
  return [cost, tokens];
}
