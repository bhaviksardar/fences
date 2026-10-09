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
