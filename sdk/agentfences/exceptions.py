from typing import Optional


class FencesError(Exception):
    pass


class AgentQuarantined(FencesError):
    """Raised when a governed function is called for an agent quarantined in the dashboard.
    No agent code runs; the run never starts."""
    def __init__(self, agent_name: str, detail: Optional[str] = None):
        self.agent_name = agent_name
        super().__init__(f"Agent {agent_name!r} is quarantined in Fences, so this run was not started"
                         + (f": {detail}" if detail and detail != "quarantined" else ""))


class FencesStop(FencesError):
    """
    Raised by framework integrations (e.g. FencesCallbackHandler) when a limit is crossed,
    because a framework callback can't return a result to the agent. Carries the same
    breach_type, message and system_prompt as a breached CheckpointResult.
    """
    def __init__(self, result):
        self.result = result
        self.breach_type = result.breach_type
        self.message = result.message
        self.system_prompt = result.system_prompt
        super().__init__(result.message)
