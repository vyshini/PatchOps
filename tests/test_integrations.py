"""
Tests for the Phase E integration endpoints (Slack / Jira / GitHub) and
the mock-mode /api/analyze streaming flow.

All outbound HTTP calls (Slack, Jira, GitHub) are mocked with `respx` -
these tests must never make a real network call. This also means they
double as documentation of the exact request shapes main.py sends to
each provider.
"""

import json

import httpx
import respx

from tests.conftest import get_user_id_from_headers


# ---------------------------------------------------------------------------
# /api/analyze (mock mode) - the core streaming flow, end to end
# ---------------------------------------------------------------------------
def test_analyze_mock_mode_streams_and_stores_incident(client, auth_headers):
    resp = client.post(
        "/api/analyze",
        json={"error_log": "ConnectionResetError at line 84", "environment": "Docker"},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    # SSE body: parse out the `done` event and confirm an incident_id
    # was minted, then confirm it's actually retrievable afterward.
    events = [
        json.loads(line[len("data: "):])
        for line in resp.text.split("\n\n")
        if line.startswith("data: ")
    ]
    done_events = [e for e in events if e.get("done")]
    assert len(done_events) == 1
    incident_id = done_events[0]["incident_id"]
    assert incident_id is not None

    detail_resp = client.get(f"/api/incidents/{incident_id}", headers=auth_headers)
    assert detail_resp.status_code == 200
    assert detail_resp.json()["severity"] in {"Sev1", "Sev2", "Sev3", "Sev4", "Unknown"}


def test_analyze_rejects_empty_error_log(client, auth_headers):
    resp = client.post("/api/analyze", json={"error_log": "   "}, headers=auth_headers)
    assert resp.status_code == 400


def test_analyze_rejects_oversized_payload(client, auth_headers):
    huge_log = "x" * 90_000
    resp = client.post("/api/analyze", json={"error_log": huge_log}, headers=auth_headers)
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Slack notification
# ---------------------------------------------------------------------------
@respx.mock
def test_notify_slack_without_config_returns_400(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)

    resp = client.post(f"/api/incidents/{incident_id}/notify-slack", headers=auth_headers)
    assert resp.status_code == 400
    assert "not configured" in resp.json()["detail"].lower()


@respx.mock
def test_notify_slack_via_webhook_succeeds(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)

    client.put(
        "/api/settings",
        json={"slack_webhook_url": "https://hooks.slack.com/services/T0/B0/xyz"},
        headers=auth_headers,
    )

    route = respx.post("https://hooks.slack.com/services/T0/B0/xyz").mock(
        return_value=httpx.Response(200, text="ok")
    )

    resp = client.post(f"/api/incidents/{incident_id}/notify-slack", headers=auth_headers)
    assert resp.status_code == 200
    assert route.called


@respx.mock
def test_notify_slack_via_bot_token_threads_on_second_call(client, auth_headers):
    """
    First send (no prior thread_ts) should post a standalone message and
    persist the returned `ts`. A second send for the SAME incident should
    reuse that ts as `thread_ts` in the follow-up chat.postMessage call.
    """
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)

    client.put(
        "/api/settings",
        json={"slack_bot_token": "xoxb-fake-token", "slack_channel_id": "C000000"},
        headers=auth_headers,
    )

    route = respx.post("https://slack.com/api/chat.postMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "ts": "1234.5678"})
    )

    first = client.post(f"/api/incidents/{incident_id}/notify-slack", headers=auth_headers)
    assert first.status_code == 200
    first_call_body = json.loads(route.calls[0].request.content)
    assert first_call_body.get("thread_ts") is None

    second = client.post(f"/api/incidents/{incident_id}/notify-slack", headers=auth_headers)
    assert second.status_code == 200
    second_call_body = json.loads(route.calls[1].request.content)
    assert second_call_body.get("thread_ts") == "1234.5678"


# ---------------------------------------------------------------------------
# Jira ticket creation / update
# ---------------------------------------------------------------------------
@respx.mock
def test_create_jira_ticket_without_config_returns_400(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    resp = client.post(f"/api/incidents/{incident_id}/jira", json={}, headers=auth_headers)
    assert resp.status_code == 400


@respx.mock
def test_create_jira_ticket_then_comment_on_second_call(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)

    client.put(
        "/api/settings",
        json={
            "jira_base_url": "https://team.atlassian.net",
            "jira_project_key": "OPS",
            "jira_email": "sre@team.com",
            "jira_api_token": "fake-token",
        },
        headers=auth_headers,
    )

    create_route = respx.post("https://team.atlassian.net/rest/api/3/issue").mock(
        return_value=httpx.Response(201, json={"key": "OPS-42"})
    )
    comment_route = respx.post("https://team.atlassian.net/rest/api/3/issue/OPS-42/comment").mock(
        return_value=httpx.Response(201, json={})
    )

    first = client.post(f"/api/incidents/{incident_id}/jira", json={}, headers=auth_headers)
    assert first.status_code == 200
    assert first.json()["jira_issue_key"] == "OPS-42"
    assert create_route.called
    assert not comment_route.called

    # Second call for the SAME incident must comment, not re-create.
    second = client.post(f"/api/incidents/{incident_id}/jira", json={}, headers=auth_headers)
    assert second.status_code == 200
    assert second.json()["jira_issue_key"] == "OPS-42"
    assert comment_route.called
    assert create_route.call_count == 1  # still only ever called once


@respx.mock
def test_create_jira_ticket_surfaces_jira_rejection_as_502(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    client.put(
        "/api/settings",
        json={
            "jira_base_url": "https://team.atlassian.net",
            "jira_project_key": "OPS",
            "jira_email": "sre@team.com",
            "jira_api_token": "fake-token",
        },
        headers=auth_headers,
    )
    respx.post("https://team.atlassian.net/rest/api/3/issue").mock(
        return_value=httpx.Response(400, text="Project key is invalid")
    )

    resp = client.post(f"/api/incidents/{incident_id}/jira", json={}, headers=auth_headers)
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# GitHub PR scaffold
# ---------------------------------------------------------------------------
@respx.mock
def test_create_github_pr_without_config_returns_400(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    resp = client.post(f"/api/incidents/{incident_id}/github-pr", json={}, headers=auth_headers)
    assert resp.status_code == 400


@respx.mock
def test_create_github_pr_full_flow(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    client.put(
        "/api/settings",
        json={"github_repo": "acme/widgets", "github_token": "fake-gh-token"},
        headers=auth_headers,
    )

    respx.get("https://api.github.com/repos/acme/widgets").mock(
        return_value=httpx.Response(200, json={"default_branch": "main"})
    )
    respx.get("https://api.github.com/repos/acme/widgets/git/ref/heads/main").mock(
        return_value=httpx.Response(200, json={"object": {"sha": "abc123"}})
    )
    respx.post("https://api.github.com/repos/acme/widgets/git/refs").mock(
        return_value=httpx.Response(201, json={})
    )
    respx.put(f"https://api.github.com/repos/acme/widgets/contents/incident-reports/incident-{incident_id}.md").mock(
        return_value=httpx.Response(201, json={})
    )
    pr_route = respx.post("https://api.github.com/repos/acme/widgets/pulls").mock(
        return_value=httpx.Response(201, json={"html_url": "https://github.com/acme/widgets/pull/7"})
    )

    resp = client.post(f"/api/incidents/{incident_id}/github-pr", json={}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["pr_url"] == "https://github.com/acme/widgets/pull/7"
    assert resp.json()["already_existed"] is False
    assert pr_route.called


@respx.mock
def test_create_github_pr_second_call_is_noop_when_pr_exists(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    client.put(
        "/api/settings",
        json={"github_repo": "acme/widgets", "github_token": "fake-gh-token"},
        headers=auth_headers,
    )

    respx.get("https://api.github.com/repos/acme/widgets").mock(
        return_value=httpx.Response(200, json={"default_branch": "main"})
    )
    respx.get("https://api.github.com/repos/acme/widgets/git/ref/heads/main").mock(
        return_value=httpx.Response(200, json={"object": {"sha": "abc123"}})
    )
    respx.post("https://api.github.com/repos/acme/widgets/git/refs").mock(
        return_value=httpx.Response(201, json={})
    )
    respx.put(f"https://api.github.com/repos/acme/widgets/contents/incident-reports/incident-{incident_id}.md").mock(
        return_value=httpx.Response(201, json={})
    )
    pr_route = respx.post("https://api.github.com/repos/acme/widgets/pulls").mock(
        return_value=httpx.Response(201, json={"html_url": "https://github.com/acme/widgets/pull/7"})
    )

    first = client.post(f"/api/incidents/{incident_id}/github-pr", json={}, headers=auth_headers)
    assert first.status_code == 200
    assert pr_route.call_count == 1

    # Second call: PR URL already stored on the incident - must short
    # circuit as a no-op rather than hitting GitHub again.
    second = client.post(f"/api/incidents/{incident_id}/github-pr", json={}, headers=auth_headers)
    assert second.status_code == 200
    assert second.json()["already_existed"] is True
    assert pr_route.call_count == 1  # unchanged - no second PR call made


@respx.mock
def test_create_github_pr_surfaces_github_rejection_as_502(client, auth_headers):
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    client.put(
        "/api/settings",
        json={"github_repo": "acme/widgets", "github_token": "fake-gh-token"},
        headers=auth_headers,
    )
    respx.get("https://api.github.com/repos/acme/widgets").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    resp = client.post(f"/api/incidents/{incident_id}/github-pr", json={}, headers=auth_headers)
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Generic outbound webhook (fire-and-forget, must never break the caller)
# ---------------------------------------------------------------------------
@respx.mock
def test_status_change_still_succeeds_when_generic_webhook_endpoint_is_down(client, auth_headers):
    """
    The generic webhook dispatch is fire-and-forget on a background task
    and must never cause the triggering request (a status update here)
    to fail, even if the configured webhook URL is unreachable.
    """
    owner_id = get_user_id_from_headers(client, auth_headers)
    from tests.conftest import create_incident_via_store

    incident_id = create_incident_via_store(user_id=owner_id)
    client.put(
        "/api/settings",
        json={"generic_webhook_url": "https://dead-endpoint.example.com/hook"},
        headers=auth_headers,
    )
    respx.post("https://dead-endpoint.example.com/hook").mock(side_effect=httpx.ConnectError("refused"))

    resp = client.patch(
        f"/api/incidents/{incident_id}/status",
        json={"status": "Resolved"},
        headers=auth_headers,
    )
    assert resp.status_code == 200