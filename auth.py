"""
auth.py
-------
Authentication layer for PatchOps: password hashing, JWT
issuance/validation, user registration/login, and the FastAPI dependency
that protects routes.

Kept intentionally simple and stateless:
  - Passwords hashed with bcrypt directly (not passlib - passlib's bcrypt
    backend has a history of version-compatibility issues; the bcrypt
    library alone is simpler and just as correct for this use case).
  - Sessions are stateless JWTs, not server-side session storage - no
    extra table/cache needed, and it scales cleanly if this ever moved
    to multiple replicas (a session store wouldn't, without adding Redis
    or similar).
  - Users live in the same SQLite file as incidents (via incident_store's
    DB_PATH) rather than a separate database - there's no operational
    reason to split them for a project at this scale, and it keeps
    deployment to a single file.

Phase E adds four more integration-settings columns (managed in
incident_store.init_db's ALTER TABLE guards): slack_bot_token and
slack_channel_id (for threaded Slack notifications via chat.postMessage,
as an alternative to a plain Incoming Webhook), and generic_webhook_url /
generic_webhook_events (for the outbound "notify any endpoint on incident
events" integration).
"""

import os
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

import incident_store

logger = logging.getLogger("patchops")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# In production this MUST be set via environment variable and kept secret -
# anyone with this key can forge valid tokens for any user. If it's not
# set, we generate a random one at startup so the app still runs (useful
# for local dev), but log a loud warning: a random key means every server
# restart invalidates all existing sessions, which is a footgun in any
# real deployment.
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
if not SECRET_KEY:
    if APP_ENV in {"production", "prod"}:
        raise RuntimeError(
            "JWT_SECRET_KEY must be set when APP_ENV is production."
        )
    import secrets

    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY is not set - using a randomly generated key for this "
        "process. All active sessions will be invalidated on restart. Set "
        "JWT_SECRET_KEY in your .env for any real deployment."
    )
elif len(SECRET_KEY) < 32:
    logger.warning(
        "JWT_SECRET_KEY is shorter than 32 characters. "
        "Use a longer random secret for stronger token signing security."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
    raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be a positive integer.")

bearer_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# REQUEST/RESPONSE SCHEMAS
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


# ---------------------------------------------------------------------------
# PASSWORD HASHING
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """
    Hashes a plaintext password with bcrypt. bcrypt has a 72-byte input
    limit - passwords longer than that are silently truncated by the
    algorithm itself, which is a known bcrypt property, not a bug here.
    """
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks a plaintext password against a stored bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# ---------------------------------------------------------------------------
# JWT ISSUANCE & VALIDATION
# ---------------------------------------------------------------------------
def create_access_token(user_id: int, username: str) -> str:
    """
    Issues a signed JWT containing the user's id and username, with a
    fixed expiry. `sub` (subject) is the standard JWT claim name for "who
    this token is about" - using it keeps the token compatible with any
    standard JWT tooling/debugger if you ever need to inspect one.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "user_id": user_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Validates and decodes a JWT. Raises a 401 HTTPException (not a bare
    exception) on any failure, so route handlers using this via Depends()
    automatically return a clean 401 to the client instead of a 500.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency that protects a route: extracts the Bearer token,
    validates it, and returns {"user_id": ..., "username": ...}. Add
    `current_user: dict = Depends(get_current_user)` to any endpoint's
    signature to require a valid logged-in user.
    """
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("user_id")
    username = payload.get("sub")
    if user_id is None or username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    return {"user_id": user_id, "username": username}


# ---------------------------------------------------------------------------
# USER CRUD
# ---------------------------------------------------------------------------
def create_user(username: str, password: str, email: Optional[str] = None) -> int:
    """
    Creates a new user with a bcrypt-hashed password. Raises ValueError
    on duplicate username (caught by the caller and turned into a clean
    409 HTTP response) rather than letting a raw sqlite3.IntegrityError
    leak up as a 500.
    """
    conn = sqlite3.connect(incident_store.DB_PATH)
    try:
        hashed = hash_password(password)
        cursor = conn.execute(
            "INSERT INTO users (username, hashed_password, email) VALUES (?, ?, ?)",
            (username, hashed, email),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = sqlite3.connect(incident_store.DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, username, hashed_password FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"id": row[0], "username": row[1], "hashed_password": row[2]}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Returns the user dict if credentials are valid, else None. Caller
    is responsible for turning a None into a 401."""
    user = get_user_by_username(username)
    if user is None:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ---------------------------------------------------------------------------
# USER SETTINGS (integrations)
# ---------------------------------------------------------------------------
# Fields that update_integration_settings() is allowed to write, and that
# get_integration_settings() reads back - one list/tuple pairing so a new
# integration field added in the future only needs to show up here (plus
# a column in incident_store.init_db) to be wired through the whole path.
_SETTINGS_COLUMNS = (
    "slack_webhook_url",
    "slack_bot_token",
    "slack_channel_id",
    "jira_base_url",
    "jira_project_key",
    "jira_email",
    "jira_api_token",
    "github_repo",
    "github_token",
    "generic_webhook_url",
    "generic_webhook_events",
)


def get_integration_settings(user_id: int) -> dict:
    """
    Returns all integration settings for server-side use, including tokens.
    Keep this for backend-only paths; never send its raw result directly to
    the browser.
    """
    conn = sqlite3.connect(incident_store.DB_PATH)
    try:
        columns_sql = ", ".join(_SETTINGS_COLUMNS)
        row = conn.execute(
            # columns_sql is built from the hardcoded _SETTINGS_COLUMNS
            # tuple above, never from user input; the actual value
            # (user_id) is still passed as a `?` parameter below.
            f"SELECT {columns_sql} FROM users WHERE id = ?",  # nosec B608
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {column: ("all" if column == "generic_webhook_events" else "") for column in _SETTINGS_COLUMNS}

    result = {column: (row[i] or "") for i, column in enumerate(_SETTINGS_COLUMNS)}
    if not result.get("generic_webhook_events"):
        result["generic_webhook_events"] = "all"
    return result


def get_integration_settings_for_api(user_id: int) -> dict:
    """
    Returns browser-safe integration settings. Secrets (API tokens, bot
    tokens) are never returned in full, only boolean "configured" flags -
    the settings UI shows "connected"/"not connected" rather than ever
    re-displaying a stored secret.
    """
    settings = get_integration_settings(user_id)
    return {
        "slack_webhook_url": settings["slack_webhook_url"],
        "slack_bot_token_configured": bool(settings["slack_bot_token"]),
        "slack_channel_id": settings["slack_channel_id"],
        "jira_base_url": settings["jira_base_url"],
        "jira_project_key": settings["jira_project_key"],
        "jira_email": settings["jira_email"],
        "jira_api_token_configured": bool(settings["jira_api_token"]),
        "github_repo": settings["github_repo"],
        "github_token_configured": bool(settings["github_token"]),
        "generic_webhook_url": settings["generic_webhook_url"],
        "generic_webhook_events": settings["generic_webhook_events"] or "all",
    }


def update_integration_settings(user_id: int, updates: dict) -> None:
    """
    Updates integration settings for a user. Empty-string values clear
    fields. Only keys present in `updates` AND in _SETTINGS_COLUMNS are
    written - anything else (e.g. a stray "id" key) is silently ignored
    rather than raising, since callers pass a Pydantic model's .dict()
    wholesale.
    """
    update_keys = [key for key in updates.keys() if key in _SETTINGS_COLUMNS]
    if not update_keys:
        return

    assignments = ", ".join(f"{key} = ?" for key in update_keys)
    params = [
        (updates[key].strip() if isinstance(updates[key], str) else updates[key]) or None
        for key in update_keys
    ]
    params.append(user_id)

    conn = sqlite3.connect(incident_store.DB_PATH)
    try:
        conn.execute(
            # `assignments` only ever contains keys already filtered
            # against the _SETTINGS_COLUMNS allowlist above (update_keys),
            # never raw user input; actual values are still passed
            # positionally as `?` parameters below.
            f"UPDATE users SET {assignments} WHERE id = ?",  # nosec B608
            params,
        )
        conn.commit()
    finally:
        conn.close()


def get_webhook_url(user_id: int) -> Optional[str]:
    """Backward-compatible helper for Slack-webhook-only callers."""
    value = get_integration_settings(user_id).get("slack_webhook_url", "")
    return value or None


def set_webhook_url(user_id: int, webhook_url: str) -> None:
    """Backward-compatible helper for Slack-webhook-only callers."""
    update_integration_settings(user_id, {"slack_webhook_url": webhook_url})
