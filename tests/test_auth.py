"""
Unit tests for auth.py - password hashing, JWT issuance/validation, and
the integration-settings helpers (including the "blank secret means
leave unchanged" contract that main.py's PUT /api/settings relies on).
"""

import jwt
import pytest
from fastapi import HTTPException

import auth


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def test_hash_and_verify_password_roundtrip():
    hashed = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", hashed)


def test_verify_password_rejects_wrong_password():
    hashed = auth.hash_password("correct horse battery staple")
    assert not auth.verify_password("wrong password", hashed)


def test_hash_password_produces_different_hash_each_time():
    # bcrypt salts each hash, so re-hashing the same password twice must
    # never produce the same stored value - otherwise salting is broken.
    h1 = auth.hash_password("same-password")
    h2 = auth.hash_password("same-password")
    assert h1 != h2


# ---------------------------------------------------------------------------
# JWT issuance / validation
# ---------------------------------------------------------------------------
def test_create_and_decode_access_token_roundtrip():
    token = auth.create_access_token(user_id=42, username="alice")
    payload = auth.decode_access_token(token)
    assert payload["user_id"] == 42
    assert payload["sub"] == "alice"


def test_decode_access_token_rejects_garbage_token():
    with pytest.raises(HTTPException) as exc_info:
        auth.decode_access_token("not.a.valid.jwt")
    assert exc_info.value.status_code == 401


def test_decode_access_token_rejects_expired_token(monkeypatch):
    # Issue a token that expired one second ago by hand, bypassing the
    # normal "expires N minutes from now" path.
    from datetime import datetime, timedelta, timezone

    expired_payload = {
        "sub": "alice",
        "user_id": 1,
        "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    expired_token = jwt.encode(expired_payload, auth.SECRET_KEY, algorithm=auth.ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        auth.decode_access_token(expired_token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_decode_access_token_rejects_token_signed_with_wrong_key():
    forged = jwt.encode({"sub": "eve", "user_id": 999}, "a-completely-different-secret", algorithm=auth.ALGORITHM)
    with pytest.raises(HTTPException):
        auth.decode_access_token(forged)


# ---------------------------------------------------------------------------
# User CRUD (needs isolated_db)
# ---------------------------------------------------------------------------
def test_create_user_and_authenticate(isolated_db):
    user_id = auth.create_user("carol", "hunter2hunter2")
    assert isinstance(user_id, int)

    authenticated = auth.authenticate_user("carol", "hunter2hunter2")
    assert authenticated is not None
    assert authenticated["username"] == "carol"


def test_authenticate_user_wrong_password_returns_none(isolated_db):
    auth.create_user("carol", "hunter2hunter2")
    assert auth.authenticate_user("carol", "wrong-password") is None


def test_authenticate_user_unknown_username_returns_none(isolated_db):
    assert auth.authenticate_user("nobody", "whatever") is None


def test_create_user_rejects_duplicate_username(isolated_db):
    auth.create_user("carol", "hunter2hunter2")
    with pytest.raises(ValueError):
        auth.create_user("carol", "different-password")


# ---------------------------------------------------------------------------
# Integration settings - the "blank secret = leave unchanged" contract
# ---------------------------------------------------------------------------
def test_update_integration_settings_persists_secret(isolated_db):
    user_id = auth.create_user("dave", "password123")
    auth.update_integration_settings(user_id, {"jira_api_token": "secret-token-abc"})

    settings = auth.get_integration_settings(user_id)
    assert settings["jira_api_token"] == "secret-token-abc"

    api_view = auth.get_integration_settings_for_api(user_id)
    assert "jira_api_token" not in api_view  # never exposed raw
    assert api_view["jira_api_token_configured"] is True


def test_update_integration_settings_only_touches_provided_keys(isolated_db):
    """
    Calling update_integration_settings with a dict that OMITS a secret
    key must not clear that secret - this is the low-level guarantee
    that main.py's endpoint relies on when it deliberately excludes
    blank secret fields from the update dict it builds.
    """
    user_id = auth.create_user("erin", "password123")
    auth.update_integration_settings(user_id, {"jira_api_token": "keep-me"})
    auth.update_integration_settings(user_id, {"slack_webhook_url": "https://hooks.slack.com/services/x"})

    settings = auth.get_integration_settings(user_id)
    assert settings["jira_api_token"] == "keep-me"
    assert settings["slack_webhook_url"] == "https://hooks.slack.com/services/x"


def test_get_integration_settings_defaults_for_unknown_user(isolated_db):
    settings = auth.get_integration_settings(user_id=999999)
    assert settings["slack_webhook_url"] == ""
    assert settings["generic_webhook_events"] == "all"