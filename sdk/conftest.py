"""
Shared pytest fixtures for the Fences test suite.

Tests run against a local backend by default (FENCES_TEST_ENDPOINT env var
to override, e.g. to point at a staging server). Each test gets a fresh
API key generated via the admin endpoint — no hardcoded keys committed
to the repo, and no risk of tests stepping on each other's data.
"""
import os
import pytest
import requests
import fences

BASE = os.environ.get("FENCES_TEST_ENDPOINT", "http://localhost:8000")
ADMIN_PASSWORD = os.environ.get("FENCES_TEST_ADMIN_PASSWORD", "test")


@pytest.fixture(autouse=True)
def fresh_fences_key():
    """
    Runs before every test: generates a brand new API key, points the
    fences SDK at it, and yields control to the test. Each test gets
    its own isolated key so test runs don't interfere with each other.
    """
    resp = requests.post(
        f"{BASE}/admin/keys/create",
        headers={"X-Admin-Password": ADMIN_PASSWORD},
        json={"label": "pytest"},
    )
    resp.raise_for_status()
    api_key = resp.json()["key"]

    fences.init(api_key=api_key, endpoint=BASE)
    yield api_key


@pytest.fixture
def api_base():
    return BASE