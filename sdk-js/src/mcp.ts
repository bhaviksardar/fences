import { getActiveRun } from "./run.js";
import { argsSummary, callAsync } from "./tools.js";

type Args = Record<string, unknown>;
interface CallToolClient { callTool(params: { name: string; arguments?: Args }, ...rest: any[]): Promise<any> }  // eslint-disable-line @typescript-eslint/no-explicit-any

export interface WrapOptions {
  /** Tool names to check with Fences before each call, or true for every tool. */
  guard?: boolean | string[];
  /** Per tool, what a call costs the run: a number or a function of its arguments. Such a tool is guarded. */
  costUsd?: Record<string, number | ((args: Args) => number)>;
  /** Tool names a person approves from the dashboard first. Such a tool is guarded. */
  approve?: string[];
}

/** An MCP tool answered with isError: recorded as the call's error, the result still goes back. */
export class MCPToolError extends Error {
  constructor(readonly result: { content?: { text?: string }[] }) {
    super((result.content ?? []).map((c) => c.text ?? "").join(" ").trim() || "isError");
    this.name = "MCPToolError";
  }
}

/**
 * Record every tool call an MCP client makes, and guard the ones you name:
 *
 *     mcp.wrap(client, { guard: ["refund", "send_email"], costUsd: { refund: (a) => a.amount as number } });
 *
 * Patches client.callTool in place and returns the client. Each call inside a governed run
 * becomes a tool span, with no tool() per tool. A result with isError counts as a failed call.
 * Guarded tools go through the same check and settle as tool(fn, { guard: true }); a refused
 * one throws ToolBlocked and never reaches the MCP server. Outside a governed run calls pass straight through.
 */
export function wrap<C extends CallToolClient>(client: C, options: WrapOptions = {}): C {
  const original = client.callTool.bind(client), costs = options.costUsd ?? {}, approve = new Set(options.approve);
  const named = new Set(Array.isArray(options.guard) ? options.guard : []);
  client.callTool = async (params, ...rest) => {
    const run = getActiveRun();
    if (!run) return original(params, ...rest);
    const { name } = params, args = params.arguments ?? {}, c = Object.hasOwn(costs, name) ? costs[name] : undefined;
    const guarded = options.guard === true || named.has(name) || c !== undefined || approve.has(name);
    try {
      return await callAsync(run, name, argsSummary([args]), guarded ? () => (typeof c === "function" ? c(args) : c) ?? 0 : null,
                             approve.has(name), async () => {
                               const result = await original(params, ...rest);
                               if (result?.isError) throw new MCPToolError(result);
                               return result;
                             }, Object.keys(args).join(", "));
    } catch (e) {
      if (e instanceof MCPToolError) return e.result;
      throw e;
    }
  };
  return client;
}
