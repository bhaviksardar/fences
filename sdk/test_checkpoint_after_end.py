import requests

API_KEY = "ag-dev-key"
BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

run_id = "finished-run-test-001"

requests.post(f"{BASE}/api/runs/start", json={
    "run_id": run_id, "agent_name": "agent_a", "budget_usd": 1.0
}, headers=HEADERS)

# End the run normally
end_resp = requests.post(f"{BASE}/api/runs/{run_id}/end", json={
    "status": "success", "error": None
}, headers=HEADERS)
print("End run:", end_resp.status_code, end_resp.json())

# Now try to sneak in more spend after it's already finished
checkpoint_resp = requests.post(f"{BASE}/api/runs/{run_id}/checkpoint", json={
    "cost_delta_usd": 0.50
}, headers=HEADERS)
print("Checkpoint after end:", checkpoint_resp.status_code, checkpoint_resp.json())

if checkpoint_resp.status_code == 409:
    print("PASSED: checkpoint on finished run correctly rejected")
else:
    print("FAILED: a finished run accepted more spend — this should be impossible")