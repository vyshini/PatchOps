"""
API-level tests against the real FastAPI app (via TestClient), covering:
  - registration/login/me
  - the IDOR protections on every /api/incidents/{id}... route (a user
    must never be able to read or mutate another user's incident)
  - status/outcome workflows
  - the PUT /api/settings "blank secret leaves it unchanged" contract,
    exercised through the actual HTTP endpoint (not just auth.py directly)
"""

from tests.conftest import create_incident_via_store, get_user_id_from_headers


# ---------------------------------------------------------------------------
# Registration / login / me
# ---------------------------------------------------------------------------
def test_register_then_me_returns_username(client):
    resp = client.post("/api/register", json={"username": "frank", "password": "password123"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    me_resp = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "frank"


def test_register_duplicate_username_returns_409(client):
    client.post("/api/register", json={"username": "frank", "password": "password123"})
    resp = client.post("/api/register", json={"username": "frank", "password": "different123"})
    assert resp.status_code == 409


def test_register_short_password_returns_400(client):
    resp = client.post("/api/register", json={"username": "gina", "password": "short"})
    assert resp.status_code == 400


def test_login_wrong_password_returns_401(client):
    client.post("/api/register", json={"username": "frank", "password": "password123"})
    resp = client.post("/api/login", json={"username": "frank", "password": "wrong-password"})
    assert resp.status_code == 401


def test_login_unknown_username_returns_401_same_as_wrong_password(client):
    """
    Deliberately the SAME error for "no such user" and "wrong password" -
    verifies main.py doesn't leak which usernames exist.
    """
    resp = client.post("/api/login", json={"username": "ghost", "password": "whatever123"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password."


def test_protected_endpoint_without_token_returns_401_or_403(client):
    # HTTPBearer returns 403 when no Authorization header is present at
    # all (vs 401 for a present-but-invalid token) - either is an
    # acceptable "you're not getting in" signal here.
    resp = client.get("/api/incidents")
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# IDOR protection across every incident-scoped route
# ---------------------------------------------------------------------------
def test_user_cannot_read_another_users_incident(client, auth_headers, second_user_auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.get(f"/api/incidents/{incident_id}", headers=second_user_auth_headers)
    assert resp.status_code == 404  # same 404 as "doesn't exist" - no enumeration signal


def test_user_cannot_update_status_of_another_users_incident(client, auth_headers, second_user_auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.patch(
        f"/api/incidents/{incident_id}/status",
        json={"status": "Resolved"},
        headers=second_user_auth_headers,
    )
    assert resp.status_code == 404

    # And the owner's incident must be untouched.
    owner_view = client.get(f"/api/incidents/{incident_id}", headers=auth_headers)
    assert owner_view.json()["status"] == "Open"


def test_user_cannot_record_outcome_on_another_users_incident(client, auth_headers, second_user_auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.post(
        f"/api/incidents/{incident_id}/outcome",
        json={"outcome": "worked", "note": "nice try"},
        headers=second_user_auth_headers,
    )
    assert resp.status_code == 404


def test_user_incident_list_excludes_other_users_incidents(client, auth_headers, second_user_auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    create_incident_via_store(user_id=owner_id)

    resp = client.get("/api/incidents", headers=second_user_auth_headers)
    assert resp.status_code == 200
    assert resp.json()["total_count"] == 0


# ---------------------------------------------------------------------------
# Status / outcome workflows
# ---------------------------------------------------------------------------
def test_update_status_happy_path(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.patch(
        f"/api/incidents/{incident_id}/status",
        json={"status": "Investigating"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "Investigating"


def test_update_status_rejects_invalid_value(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.patch(
        f"/api/incidents/{incident_id}/status",
        json={"status": "Escalated"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_record_outcome_updates_historical_success_rate(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.post(
        f"/api/incidents/{incident_id}/outcome",
        json={"outcome": "worked", "note": "restarted the pod"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["summary"]["historical_success_rate"] == 1.0


def test_record_outcome_rejects_invalid_value(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.post(
        f"/api/incidents/{incident_id}/outcome",
        json={"outcome": "sort-of-worked"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_record_outcome_on_nonexistent_incident_returns_404(client, auth_headers):
    resp = client.post(
        "/api/incidents/999999/outcome",
        json={"outcome": "worked"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Settings endpoint - blank-secret-preserves-existing contract, end to end
# ---------------------------------------------------------------------------
def test_settings_roundtrip_never_exposes_raw_secret(client, auth_headers):
    resp = client.put(
        "/api/settings",
        json={"jira_api_token": "super-secret-token", "jira_base_url": "https://team.atlassian.net"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "jira_api_token" not in body
    assert body["jira_api_token_configured"] is True
    assert body["jira_base_url"] == "https://team.atlassian.net"


def test_settings_blank_secret_field_does_not_clear_existing_token(client, auth_headers):
    # First save sets a real Jira token.
    client.put("/api/settings", json={"jira_api_token": "super-secret-token"}, headers=auth_headers)

    # Second save omits/blanks the token field but changes something
    # else - the previously-saved token must survive untouched.
    resp = client.put(
        "/api/settings",
        json={"jira_api_token": "", "jira_project_key": "OPS"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jira_api_token_configured"] is True
    assert body["jira_project_key"] == "OPS"


def test_settings_rejects_non_https_url(client, auth_headers):
    resp = client.put(
        "/api/settings",
        json={"slack_webhook_url": "http://insecure.example.com/hook"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_settings_rejects_malformed_github_repo(client, auth_headers):
    resp = client.put(
        "/api/settings",
        json={"github_repo": "not-owner-slash-repo"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_settings_rejects_unknown_webhook_event(client, auth_headers):
    resp = client.put(
        "/api/settings",
        json={"generic_webhook_url": "https://example.com/hook", "generic_webhook_events": "made_up_event"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Filtering / pagination sanity checks
# ---------------------------------------------------------------------------
def test_list_incidents_filters_by_environment(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    create_incident_via_store(user_id=owner_id, environment="Docker")
    create_incident_via_store(user_id=owner_id, environment="Kubernetes")

    resp = client.get("/api/incidents?environment=Docker", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 1
    assert data["incidents"][0]["environment"] == "Docker"


def test_list_incidents_rejects_invalid_severity_filter(client, auth_headers):
    resp = client.get("/api/incidents?severity=NotARealSeverity", headers=auth_headers)
    assert resp.status_code == 400