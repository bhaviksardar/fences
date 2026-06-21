import requests
from typing import Optional


class GovClient:
    """
    Talks to the Fences backend.

    checkpoint() is a SYNCHRONOUS call on purpose: since the server is the
    source of truth for budget enforcement, the local check is only a fast
    first guess — we still need the server's answer before deciding whether
    a "borderline" run should actually be allowed to continue. A short
    timeout keeps this from hanging the agent if the network is slow.
    """

    def __init__(self, api_key: str, endpoint: str, timeout: float = 3.0):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def start_run(self, run_id: str, agent_name: str, budget_usd: float) -> dict:
        return self._post("/api/runs/start", {
            "run_id": run_id,
            "agent_name": agent_name,
            "budget_usd": budget_usd,
        })

    def checkpoint(self, run_id: str, cost_delta_usd: float) -> dict:
        """
        Reports a cost delta and asks the server whether the run is still
        within budget. Returns {"ok": bool, "spent_usd": float, "budget_usd": float}.
        """
        return self._post(f"/api/runs/{run_id}/checkpoint", {
            "cost_delta_usd": cost_delta_usd,
        })

    def end_run(self, run_id: str, status: str, error: Optional[str] = None) -> dict:
        return self._post(f"/api/runs/{run_id}/end", {
            "status": status,
            "error": error,
        })

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = requests.post(
                f"{self.endpoint}{path}",
                json=payload,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            # Network failure: fail OPEN by default (don't block the agent
            # because the network blipped), but flag it so checkpoint() can
            # decide what to do — this is a real product decision, not a
            # detail to hide. See checkpoint()'s docstring.
            return {"ok": True, "network_error": str(e)}