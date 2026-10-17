"""HTTP calls to a Fences server, as specified in PROTOCOL.md. No call raises, except a rejected key at run start."""
import logging
import random
import threading
import time
from typing import Optional

import requests

from . import __version__
from .exceptions import FencesAuthError

log = logging.getLogger("agentfences")

FAILURES_TO_OPEN = 3  # failed calls in a row before the breaker opens
OPEN_S = 30.0         # how long calls are skipped once it's open (with jitter), before one goes through to test


class Breaker:
    """
    A circuit breaker: after FAILURES_TO_OPEN failed calls in a row (no connection, a timeout,
    a 5xx, or a 429 on anything but the heartbeat), skip calls for about OPEN_S seconds, then let one through to test the server.
    While it's open a call fails at once instead of after the timeout: checkpoints are enforced
    locally and guarded tools are blocked, without adding seconds to every step.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.failures, self.open_until, self.probing = 0, 0.0, False

    def allow(self) -> bool:
        with self._lock:
            if self.failures < FAILURES_TO_OPEN:
                return True
            if self.probing or time.monotonic() < self.open_until:
                return False
            self.probing = True  # one test call at a time
            return True

    def record(self, ok: bool):
        with self._lock:
            was_open, self.probing = self.failures >= FAILURES_TO_OPEN, False
            if ok:
                self.failures = 0
                if was_open:
                    log.warning("agentfences: Fences is reachable again")
                return
            self.failures += 1
            if self.failures >= FAILURES_TO_OPEN:
                # jitter, so SDKs cut off by the same outage don't all come back at once
                self.open_until = time.monotonic() + OPEN_S * random.uniform(0.75, 1.25)
                if not was_open:
                    log.warning("agentfences: Fences failed %d calls in a row; skipping calls for about %ds "
                                "(limits enforced locally, guarded tools blocked)", self.failures, OPEN_S)


class GovClient:
    def __init__(self, api_key: str, endpoint: str, timeout: float = 3.0):
        # The server uses the SDK version to warn about old SDKs and to skip commands they don't understand
        self.headers = {"X-API-Key": api_key, "X-Fences-SDK": f"python/{__version__}"}
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()  # a Session per thread: connections are reused, and Sessions aren't thread-safe
        self.breaker = Breaker()
        self.dedupes_steps = False  # the server echoed a step_id: it counts a repeated step once, so a retry is safe

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

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
        return self._post("/api/runs/start", payload, start=True)

    def checkpoint(self, run_id: str, cost_delta_usd: float, iterations: int, duration_ms: int, tokens_used: int,
                   step_id: Optional[str] = None) -> dict:
        body = {"cost_delta_usd": cost_delta_usd, "iterations": iterations, "duration_ms": duration_ms, "tokens_used": tokens_used}
        if step_id:
            body["step_id"] = step_id  # the server counts a step once, however often it's sent
        resp = self._post(f"/api/runs/{run_id}/checkpoint", body, retry=bool(step_id) and self.dedupes_steps)
        if step_id and resp.get("step_id") == step_id:
            self.dedupes_steps = True
        return resp

    def log_decision(self, run_id: str, iteration: int, reasoning: str, action: Optional[str]) -> dict:
        return self._post(f"/api/runs/{run_id}/decisions", {
            "iteration": iteration,
            "reasoning": reasoning,
            "action": action,
        })

    def log_spans(self, run_id: str, spans: list) -> dict:
        return self._post(f"/api/runs/{run_id}/spans", {"spans": spans}, optional=True)

    def heartbeat(self, runs: list, acks: Optional[list] = None) -> dict:
        """Report live runs and acknowledge commands; the reply may carry commands for them."""
        return self._post("/api/heartbeat", {"runs": runs, **({"acks": acks} if acks else {})}, optional=True)

    def request_approval(self, run_id: str, approval_id: str, reason: str, amount_usd: Optional[float],
                         timeout_s: float) -> dict:
        return self._post(f"/api/runs/{run_id}/approvals", {"approval_id": approval_id, "reason": reason,
                          "amount_usd": amount_usd, "timeout_s": timeout_s}, optional=True)

    def guard(self, run_id: str, body: dict) -> dict:
        """Check a guarded tool call before it runs ({"phase": "check"}), or settle it after ({"phase": "settle"})."""
        return self._post(f"/api/runs/{run_id}/guard", body, optional=True, retry=True)  # guard_id: safe to repeat

    def end_run(self, run_id: str, status: str, error: Optional[str] = None, exception: Optional[dict] = None,
                breach: Optional[str] = None) -> dict:
        payload = {"status": status, "error": error}
        if breach:
            payload["breach"] = breach
        if exception:
            payload["exception"] = exception
        return self._post(f"/api/runs/{run_id}/end", payload)

    def _post(self, path: str, payload: dict, optional: bool = False, start: bool = False, retry: bool = False) -> dict:
        """
        optional: an endpoint older servers may not have. A missing route returns
        {"unsupported": True}, and any other error {"ok": False, "status": ...}.
        start: a rejected key raises FencesAuthError, before any agent code runs. Later in a run
        it returns {"network_error": ..., "auth_rejected": True}: raising there would replace
        the agent's own result or exception.
        retry: a call that's safe to repeat goes once more after a connection error (not after a
        timeout, which would double the wait).
        """
        if not self.breaker.allow():
            return {"network_error": "Fences unreachable: calls paused after repeated failures"}
        try:
            try:
                resp = self._session().post(f"{self.endpoint}{path}", json=payload, headers=self.headers, timeout=self.timeout)
            except requests.ConnectionError as e:
                if not retry or isinstance(e, requests.Timeout):  # a connect timeout is both; timeouts aren't retried
                    raise
                time.sleep(random.uniform(0.1, 0.3))  # a blip: once more, after a moment
                resp = self._session().post(f"{self.endpoint}{path}", json=payload, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as e:
            self.breaker.record(False)
            return {"network_error": str(e)}
        except ValueError as e:  # a payload JSON can't encode: not the server's fault
            self.breaker.record(True)
            return {"network_error": str(e)}
        # A 429 on the heartbeat is its own budget, not an outage: it mustn't cut off checkpoints
        self.breaker.record(resp.status_code < 500 and (resp.status_code != 429 or path == "/api/heartbeat"))
        try:
            body = resp.json()
        except ValueError:  # not JSON: the status code still decides
            body = None
        is_json = isinstance(body, dict)  # a list or a bare value isn't a Fences reply either
        body = body if is_json else {}
        detail = body.get("detail", "") if isinstance(body.get("detail"), str) else ""
        if optional:
            if resp.status_code in (404, 405) and detail in ("Not Found", "Method Not Allowed"):
                return {"unsupported": True}
            return body if resp.ok else {"ok": False, "status": resp.status_code}
        if resp.status_code in (401, 403):
            if start:
                raise FencesAuthError(f"Fences API key rejected: {detail or resp.text[:200]}")
            return {"network_error": f"API key rejected (HTTP {resp.status_code})", "auth_rejected": True}
        if resp.status_code == 423:  # the agent is quarantined: the run must not start
            return {"ok": False, "quarantined": True, "detail": detail}
        if resp.status_code == 409:  # e.g. checkpoint on a run the server already fenced
            return {"ok": False, "conflict": detail}
        if not resp.ok or not is_json:  # an error, or not a Fences reply: enforce locally
            return {"network_error": f"HTTP {resp.status_code}" if not resp.ok else "response was not JSON"}
        return body