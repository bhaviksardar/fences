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
        try:
            return self._post(f"/api/runs/{run_id}/decisions", {
                "iteration": iteration,
                "reasoning": reasoning,
                "action": action,
            })
        except Exception:
            return {}

    def log_events(self, run_id: str, events: list) -> dict:
        return self._post_optional(f"/api/runs/{run_id}/events", {"events": events})

    def heartbeat(self, runs: list) -> dict:
        """Report live runs; the reply may carry commands for them."""
        return self._post_optional("/api/heartbeat", {"runs": runs})

    def request_approval(self, run_id: str, approval_id: str, reason: str, amount_usd: Optional[float]) -> dict:
        return self._post_optional(f"/api/runs/{run_id}/approvals",
                                   {"approval_id": approval_id, "reason": reason, "amount_usd": amount_usd})

    def _post_optional(self, path: str, payload: dict) -> dict:
        """POST to an endpoint older servers may not have: {"unsupported": True} if the route is missing."""
        try:
            resp = requests.post(f"{self.endpoint}{path}", json=payload, headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as e:
            return {"network_error": str(e)}
        is_json = resp.headers.get("content-type", "").startswith("application/json")
        if resp.status_code in (404, 405) and is_json and resp.json().get("detail") in ("Not Found", "Method Not Allowed"):
            return {"unsupported": True}
        if not resp.ok:
            return {"ok": False, "status": resp.status_code}
        return resp.json() if is_json else {"ok": True}

    def end_run(self, run_id: str, status: str, error: Optional[str] = None, exception: Optional[dict] = None) -> dict:
        payload = {"status": status, "error": error}
        if exception:
            payload["exception"] = exception
        return self._post(f"/api/runs/{run_id}/end", payload)

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
            if resp.status_code == 423:  # the agent is quarantined: the run must not start
                return {"ok": False, "quarantined": True, "detail": resp.json().get("detail", "")}
            if resp.status_code == 409:  # e.g. checkpoint on a run the server already fenced
                return {"ok": False, "conflict": resp.json().get("detail", "")}
            resp.raise_for_status()
            return resp.json()
        except PermissionError:
            raise
        except (requests.RequestException, ValueError) as e:
            return {"network_error": str(e)}