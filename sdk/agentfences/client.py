import requests
from typing import Optional


class GovClient:
    def __init__(self, api_key: str, endpoint: str, timeout: float = 3.0):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def start_run(self, run_id: str, agent_name: str, budget_usd: float, max_iterations: int, max_duration_ms: int, max_tokens: int,
                  context: Optional[dict] = None) -> dict:
        payload = {
            "run_id": run_id,
            "agent_name": agent_name,
            "budget_usd": budget_usd,
            "max_iterations": max_iterations,
            "max_duration_ms": max_duration_ms,
            "max_tokens": max_tokens,
        }
        if context:
            payload["context"] = context
        return self._post("/api/runs/start", payload)

    def checkpoint(self, run_id: str, cost_delta_usd: float, iterations: int, duration_ms: int, tokens_used: int) -> dict:
        return self._post(f"/api/runs/{run_id}/checkpoint", {
            "cost_delta_usd": cost_delta_usd,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "tokens_used": tokens_used,
        })

    def log_decision(self, run_id: str, iteration: int, reasoning: str, action: Optional[str]) -> dict:
        return self._post(f"/api/runs/{run_id}/decisions", {
            "iteration": iteration,
            "reasoning": reasoning,
            "action": action,
        })

    def log_events(self, run_id: str, events: list) -> dict:
        return self._post(f"/api/runs/{run_id}/events", {"events": events}, optional=True)

    def heartbeat(self, runs: list) -> dict:
        """Report live runs; the reply may carry commands for them."""
        return self._post("/api/heartbeat", {"runs": runs}, optional=True)

    def request_approval(self, run_id: str, approval_id: str, reason: str, amount_usd: Optional[float]) -> dict:
        return self._post(f"/api/runs/{run_id}/approvals",
                          {"approval_id": approval_id, "reason": reason, "amount_usd": amount_usd}, optional=True)

    def end_run(self, run_id: str, status: str, error: Optional[str] = None, exception: Optional[dict] = None) -> dict:
        payload = {"status": status, "error": error}
        if exception:
            payload["exception"] = exception
        return self._post(f"/api/runs/{run_id}/end", payload)

    def _post(self, path: str, payload: dict, optional: bool = False) -> dict:
        """
        optional: an endpoint older servers may not have. A missing route returns
        {"unsupported": True} and no error raises, not even a rejected key.
        """
        try:
            resp = requests.post(f"{self.endpoint}{path}", json=payload, headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as e:
            return {"network_error": str(e)}
        try:
            body, is_json = resp.json(), True
        except ValueError:  # not JSON: the status code still decides
            body, is_json = {}, False
        detail = body.get("detail", "") if isinstance(body, dict) else ""
        if optional:
            if resp.status_code in (404, 405) and detail in ("Not Found", "Method Not Allowed"):
                return {"unsupported": True}
            return body if resp.ok else {"ok": False, "status": resp.status_code}
        if resp.status_code in (401, 403):
            raise PermissionError(f"Fences API key rejected: {resp.text}")
        if resp.status_code == 423:  # the agent is quarantined: the run must not start
            return {"ok": False, "quarantined": True, "detail": detail}
        if resp.status_code == 409:  # e.g. checkpoint on a run the server already fenced
            return {"ok": False, "conflict": detail}
        if not resp.ok or not is_json:  # an error, or not a Fences reply: enforce locally
            return {"network_error": f"HTTP {resp.status_code}" if not resp.ok else "response was not JSON"}
        return body