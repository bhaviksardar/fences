from typing import Optional


class FencesError(Exception):
    pass


class FencesAuthError(FencesError, PermissionError):
    """The Fences server rejected the API key when a run started. Raised so a bad key is noticed, not silently ignored.
    A key revoked during a run ends it at its next checkpoint instead (breach "fences_unreachable")."""


class AgentQuarantined(FencesError):
    """Raised when a governed function is called for an agent quarantined in the dashboard.
    No agent code runs; the run never starts."""
    def __init__(self, agent_name: str, detail: Optional[str] = None):
        self.agent_name = agent_name
        super().__init__(f"Agent {agent_name!r} is quarantined in Fences, so this run was not started"
                         + (f": {detail}" if detail and detail != "quarantined" else ""))


class FencesStop(FencesError):
    """
    Raised when a run is halted from outside the agent's own checkpoint() calls: a framework
    integration at a crossed limit (a callback can't return a result to the agent), a model
    call refused by the budget precheck, or an async run stuck after a dashboard stop
    (stop_grace_s). Carries the same breach_type, message and system_prompt as a breached
    CheckpointResult.
    """
    def __init__(self, result):
        self.result = result
        self.breach_type = result.breach_type
        self.message = result.message
        self.system_prompt = result.system_prompt
        super().__init__(result.message)


class ToolBlocked(FencesError):
    """
    Raised instead of running a guarded tool (@agentfences.tool(guard=True)) that Fences
    didn't allow: the run is stopped, over budget, unapproved, or Fences couldn't be reached.
    The tool's code never ran. reason says why; breach_type is set when a limit is the cause.
    """
    def __init__(self, tool: str, reason: str, breach_type: Optional[str] = None):
        self.tool, self.reason, self.breach_type = tool, reason, breach_type
        super().__init__(f"Fences blocked {tool}: {reason}")
