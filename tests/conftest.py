"""
conftest.py
-----------
Shared pytest fixtures for the PatchOps test suite.

Key design decision: every test gets its OWN temp SQLite file, not the
real incidents.db and not a single shared :memory: DB. A single shared
:memory: DB doesn't work here because incident_store.py/auth.py each
open a fresh sqlite3.connect(DB_PATH) per call - separate connections to
":memory:" are separate, unrelated databases, so state wouldn't persist
between calls within one test. A real temp file (via pytest's `tmp_path`)
behaves exactly like production SQLite usage while still being fully
isolated and auto-cleaned per test.

We monkeypatch `incident_store.DB_PATH` directly (not just the
INCIDENT_DB_PATH env var) because DB_PATH is read once at import time
from os.getenv(...) - by the time tests run, main.py has already been
imported and incident_store.DB_PATH is already a fixed string. Every
function in incident_store.py and auth.py reads the module-level
`incident_store.DB_PATH` attribute at CALL time (not a cached copy), so
patching that attribute is what actually redirects every DB call for the
duration of the test.
"""

import os

# Must be set before `main` (and therefore `auth`, `incident_store`) is
# ever imported by any test module, since auth.py reads JWT_SECRET_KEY /
# APP_ENV at import time and main.py reads MOCK_MODE at import time.
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-do-not-use-in-prod-0123456789")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("GEMINI_API_KEY", "")

import pytest
from fastapi.testclient import TestClient

import incident_store
import main


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """
    Points incident_store.DB_PATH (and therefore every DB call made by
    auth.py and incident_store.py) at a fresh temp file for this test,
    then initializes the schema against it. Yields the path in case a
    test wants to assert directly against the file.
    """
    db_file = str(tmp_path / "test_patchops.db")
    monkeypatch.setattr(incident_store, "DB_PATH", db_file)
    incident_store.init_db()
    yield db_file


@pytest.fixture()
def client(isolated_db):
    """A TestClient bound to the real FastAPI app, with an isolated DB."""
    return TestClient(main.app)


def _register(client: TestClient, username: str, password: str = "correct-horse-battery") -> dict:
    resp = client.post(
        "/api/register",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture()
def auth_headers(client):
    """Registers a fresh user and returns ready-to-use Authorization headers."""
    token_data = _register(client, "alice")
    return {"Authorization": f"Bearer {token_data['access_token']}"}


@pytest.fixture()
def second_user_auth_headers(client):
    """A second, distinct user - for IDOR / cross-user isolation tests."""
    token_data = _register(client, "bob")
    return {"Authorization": f"Bearer {token_data['access_token']}"}


def create_incident_via_store(
    user_id: int,
    error_log: str = "ConnectionResetError at line 84",
    environment: str = "Docker",
) -> int:
    """
    Directly inserts an incident via incident_store.store_incident(),
    bypassing the LLM/mock streaming endpoint - used by tests that need
    a stored incident to exist but don't care about the analysis flow
    itself (e.g. status/outcome/IDOR tests).
    """
    fake_embedding = [0.1 * i for i in range(16)]
    analysis_text = (
        "## 1. Root Cause Analysis\n"
        "[SEVERITY: Sev3]\n\n"
        "- [HIGH-CONFIDENCE] The log shows a ConnectionResetError.\n\n"
        "## 2. Architecture Impact\n"
        "```mermaid\nflowchart LR\nA-->B\n```\n\n"
        "## 3. Remediation / Git Patch\n"
        "```diff\n--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n```\n"
    )
    return incident_store.store_incident(
        user_id=user_id,
        error_log=error_log,
        environment=environment,
        analysis_text=analysis_text,
        embedding=fake_embedding,
    )


def get_user_id_from_headers(client: TestClient, headers: dict) -> int:
    resp = client.get("/api/me", headers=headers)
    assert resp.status_code == 200
    return resp.json()["user_id"]