import type { CheckpointResult } from "./core.js";

export class FencesError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The Fences server rejected the API key. Thrown so a bad key is noticed, not silently ignored. */
export class FencesAuthError extends FencesError {}

/** A governed function was called for an agent quarantined in the dashboard. No agent code ran. */
export class AgentQuarantined extends FencesError {
  constructor(public readonly agentName: string, detail?: string) {
    super(`Agent '${agentName}' is quarantined in Fences, so this run was not started` +
      (detail && detail !== "quarantined" ? `: ${detail}` : ""));
  }
}

/**
 * Thrown instead of running a guarded tool (tool(fn, { guard: true })) that Fences didn't
 * allow: the run is stopped, over budget, unapproved or quarantined, or Fences couldn't be
 * reached. The tool's code never ran. reason says why; breachType is set when a limit is the cause.
 */
export class ToolBlocked extends FencesError {
  constructor(public readonly tool: string, public readonly reason: string, public readonly breachType: string | null = null) {
    super(`Fences blocked ${tool}: ${reason}`);
  }
}

/**
 * Thrown when a run is halted from outside the agent's own code: a governed call stuck after a
 * dashboard stop (see stopGraceS). Carries the same fields as a breached CheckpointResult.
 */
export class FencesStop extends FencesError {
  readonly breachType: string | null;
  readonly systemPrompt: string;
  constructor(public readonly result: CheckpointResult) {
    super(result.message);
    this.breachType = result.breachType;
    this.systemPrompt = result.systemPrompt;
  }
}
