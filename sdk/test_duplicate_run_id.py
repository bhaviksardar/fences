import requests

API_KEY = "ag-dev-key"
BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": API_KEY}

run_id = "dup-test-001"

r1 = requests.post(f"{BASE}/api/runs/start", json={
    "run_id": run_id, "agent_name": "agent_a", "budget_usd": 0.50
}, headers=HEADERS)
print("First start:", r1.status_code, r1.json())

r2 = requests.post(f"{BASE}/api/runs/start", json={
    "run_id": run_id, "agent_name": "agent_b", "budget_usd": 99.0  # trying to overwrite with a huge budget
}, headers=HEADERS)
print("Duplicate start:", r2.status_code, r2.json())

if r2.status_code == 409:
    print("PASSED: duplicate run_id correctly rejected")
else:
    print("FAILED: duplicate run_id was NOT rejected — budget could be silently overwritten")