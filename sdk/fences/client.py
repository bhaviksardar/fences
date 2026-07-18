import requests
from typing import Optional


class GovClient:
    def __init__(self, api_key: str, endpoint: str, timeout: float = 3.0):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def start_run(self, run_id: str, agent_name: str, budget_usd: float, max_iterations: int, max_duration_ms: int) -> dict:
        return self._post("/api/runs/start", {
            "run_id": run_id,
            "agent_name": agent_name,
            "budget_usd": budget_usd,
            "max_iterations": max_iterations,
            "max_duration_ms": max_duration_ms,
        })

    def checkpoint(self, run_id: str, cost_delta_usd: float, iterations: int, duration_ms: int) -> dict:
        return self._post(f"/api/runs/{run_id}/checkpoint", {
            "cost_delta_usd": cost_delta_usd,
            "iterations": iterations,
            "duration_ms": duration_ms,
        })

    def log_decision(self, run_id: str, iteration: int, reasoning: str, action: Optional[str]) -> dict:
        try:
            return self._post(f"/api/runs/{run_id}/decisions", {
                "iteration": iteration,
                "reasoning": reasoning,
                "action": action,
            })
        except Exception:
            return {}

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
            if resp.status_code in (401, 403):
                raise PermissionError(f"Fences API key rejected: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except PermissionError:
            raise
        except requests.RequestException as e:
            return {"ok": True, "network_error": str(e)}