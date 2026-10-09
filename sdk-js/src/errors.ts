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
 * Thrown by framework integrations when a limit is crossed, because a framework callback
 * can't hand a result back to the agent. Carries the same fields as a breached CheckpointResult.
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
