"""
incident_store.py
------------------
Lightweight "similar past incidents" retrieval layer for the SRE AI
Copilot. This is a minimal, dependency-light form of Retrieval-Augmented
Generation (RAG):

  1. Every analyzed incident is embedded (turned into a vector that
     represents its semantic meaning) and stored in a local SQLite file.
  2. When a NEW incident comes in, we embed it too, then compare it
     against every stored embedding using cosine similarity.
  3. The best matches (if any are similar enough) are handed back to the
     caller, who injects them into the LLM prompt as extra context.

Deliberately kept simple for a single-instance deployment:
  - SQLite file on disk, not a hosted vector DB - fine at hundreds to a
    few thousand rows, which is the realistic ceiling for a demo/portfolio
    tool. If this needed to scale to millions of incidents, you'd swap
    the brute-force similarity loop below for a proper vector index
    (e.g. pgvector, FAISS) - but that would be solving a problem this
    project doesn't actually have.
  - Cosine similarity computed in plain Python/math, no numpy dependency,
    since embeddings here are only ~768 floats and we're comparing
    against at most a few thousand rows per request.

Phase E adds three integration-tracking columns on `incidents`
(jira_issue_key, github_pr_url, slack_thread_ts) so the app can remember
"we already have a Jira ticket / PR / Slack thread for this incident" and
update-in-place instead of creating duplicates on a second click.
"""

import os
import json
import sqlite3
import logging
import math
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

logger = logging.getLogger("patchops")

DB_PATH = os.getenv("INCIDENT_DB_PATH", "incidents.db")

# Gemini's dedicated embedding model - separate from the generation model
# (GEMINI_MODEL) used for the analysis/post-mortem text itself. Embedding
# models are smaller, cheaper, and purpose-built for producing vectors
# rather than prose.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Cosine similarity ranges from -1 to 1; for real embeddings of natural
# language it's almost always in the 0-1 range. Anything below this
# threshold is treated as "not actually similar" and excluded, rather
# than being force-fed to the LLM as false context. This number was
# picked conservatively (favoring precision over recall) - it's better
# to show zero matches than to inject a misleading one.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# How many top matches to return at most, even if more clear the
# threshold - keeps prompt context bounded and the UI panel readable.
MAX_MATCHES = int(os.getenv("MAX_SIMILAR_INCIDENTS", "3"))
ANALYTICS_DEFAULT_DAYS = int(os.getenv("ANALYTICS_DEFAULT_DAYS", "30"))
ANALYTICS_MAX_DAYS = int(os.getenv("ANALYTICS_MAX_DAYS", "180"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "0.80"))
OUTCOME_WEIGHT = float(os.getenv("OUTCOME_WEIGHT", "0.20"))

SORTABLE_INCIDENT_FIELDS = {
    "created_at": "created_at",
    "severity": "severity",
    "status": "status",
    "environment": "environment",
}


@dataclass
class SimilarIncident:
    """One retrieval result - a past incident plus how similar it is to
    the current one being analyzed."""
    incident_id: int
    error_log: str
    environment: str
    analysis_text: str
    created_at: str
    similarity: float
    historical_success_rate: Optional[float]
    outcome_samples: int
    ranking_score: float


def init_db() -> None:
    """
    Creates the incidents table if it doesn't exist yet. Called once at
    app startup. `embedding` is stored as a JSON-encoded list of floats -
    SQLite has no native vector type, and at this scale a JSON TEXT
    column with brute-force comparison is simpler and more transparent
    than a binary blob format.

    `user_id` ties each incident to the user who analyzed it (added for
    Day 1 auth work). For a brand-new DB it's part of the CREATE TABLE
    below; for a DB created before auth existed, the ALTER TABLE guard
    adds the column to existing installs without losing data - SQLite
    has no "ADD COLUMN IF NOT EXISTS", so we catch the "duplicate
    column" error instead, which is the standard workaround.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_log TEXT NOT NULL,
                environment TEXT NOT NULL,
                analysis_text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                user_id INTEGER,
                severity TEXT NOT NULL DEFAULT 'Unknown',
                status TEXT NOT NULL DEFAULT 'Open',
                resolved_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN user_id INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists - fine, this is the common case
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN severity TEXT NOT NULL DEFAULT 'Unknown'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN status TEXT NOT NULL DEFAULT 'Open'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN resolved_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN fingerprint TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN duplicate_of_id INTEGER")
        except sqlite3.OperationalError:
            pass
        # --- Phase E: integration-tracking columns ---------------------
        # These remember the external artifact we already created for an
        # incident, so a second click on "Create Jira Ticket" / "Open
        # GitHub PR" updates the existing thing instead of spawning a
        # duplicate ticket/PR every time.
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN jira_issue_key TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN github_pr_url TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE incidents ADD COLUMN slack_thread_ts TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN slack_webhook_url TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN jira_base_url TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN jira_project_key TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN jira_email TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN jira_api_token TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN github_repo TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN github_token TEXT")
        except sqlite3.OperationalError:
            pass
        # --- Phase E: Slack bot mode (enables threading) + generic
        # outbound webhook config. slack_webhook_url (Incoming Webhook)
        # remains supported for one-shot posts; slack_bot_token +
        # slack_channel_id unlock chat.postMessage, which returns a
        # message `ts` we can reply to in-thread for follow-up events
        # (status changes, outcome feedback) on the same incident.
        try:
            conn.execute("ALTER TABLE users ADD COLUMN slack_bot_token TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN slack_channel_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN generic_webhook_url TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "ALTER TABLE users ADD COLUMN generic_webhook_events TEXT NOT NULL DEFAULT 'all'"
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incidents_user_created "
            "ON incidents(user_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incidents_user_status "
            "ON incidents(user_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incidents_user_severity "
            "ON incidents(user_id, severity)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incidents_user_fingerprint "
            "ON incidents(user_id, fingerprint)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incident_events_incident_created "
            "ON incident_events(incident_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incident_events_user_created "
            "ON incident_events(user_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incident_outcomes_incident_created "
            "ON incident_outcomes(incident_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incident_outcomes_user_created "
            "ON incident_outcomes(user_id, created_at DESC)"
        )

        rows_to_backfill = conn.execute(
            "SELECT id, error_log FROM incidents WHERE fingerprint IS NULL"
        ).fetchall()
        for incident_id, error_log in rows_to_backfill:
            conn.execute(
                "UPDATE incidents SET fingerprint = ? WHERE id = ?",
                (build_incident_fingerprint(error_log), incident_id),
            )
        conn.commit()
    finally:
        conn.close()


# Allowed status values for an incident's workflow state, and the only
# values update_incident_status() will accept - kept as a small module-
# level constant so both the validation logic here and any place that
# needs to enumerate them (e.g. an API docs description) stay in sync.
VALID_STATUSES = ("Open", "Investigating", "Resolved", "Closed")
VALID_SEVERITIES = ("Sev1", "Sev2", "Sev3", "Sev4", "Unknown")
VALID_OUTCOMES = ("worked", "partial", "failed")
OUTCOME_SCORES = {"worked": 1.0, "partial": 0.5, "failed": 0.0}

# Event types the generic outbound webhook (Phase E) can fire for. Kept
# as a module constant so main.py's dispatcher and any future settings-UI
# "which events do you want" picker stay in sync with what's actually
# emitted, rather than each guessing at the same string literals.
WEBHOOK_EVENT_TYPES = (
    "incident_created",
    "status_changed",
    "outcome_recorded",
)


def _insert_event(
    conn: sqlite3.Connection,
    incident_id: int,
    user_id: int,
    event_type: str,
    previous_status: Optional[str] = None,
    new_status: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO incident_events (
            incident_id, user_id, event_type, previous_status, new_status, note
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (incident_id, user_id, event_type, previous_status, new_status, note),
    )


def extract_severity(analysis_text: str) -> str:
    """
    Pulls the severity classification out of the model's analysis text.
    The prompt requires the model to emit a line formatted exactly
    `[SEVERITY: SevX]` as the first line of Section 1 - this regex finds
    it regardless of exact position, so a model that doesn't perfectly
    follow the "first line" instruction still gets classified correctly
    rather than silently falling back to "Unknown".

    Returns "Unknown" (not an exception) if the tag is missing or
    malformed - severity classification is a best-effort enhancement, not
    something that should ever block storing an otherwise-successful
    analysis.
    """
    match = re.search(r"\[SEVERITY:\s*Sev([1-4])\]", analysis_text, re.IGNORECASE)
    if match:
        return f"Sev{match.group(1)}"
    return "Unknown"


def build_incident_fingerprint(error_log: str) -> str:
    """
    Builds a deterministic fingerprint for dedup/clustering by normalizing
    noisy runtime values out of the log before hashing.
    """
    normalized = error_log.lower()
    normalized = re.sub(r"0x[0-9a-f]+", "0x<hex>", normalized)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b", "<timestamp>", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_outcome_stats_for_incidents(
    conn: sqlite3.Connection,
    user_id: int,
) -> Dict[int, Dict[str, Optional[float]]]:
    rows = conn.execute(
        """
        SELECT
            incident_id,
            SUM(CASE outcome WHEN 'worked' THEN 1.0 WHEN 'partial' THEN 0.5 ELSE 0.0 END) AS score_sum,
            COUNT(*) AS sample_count
        FROM incident_outcomes
        WHERE user_id = ?
        GROUP BY incident_id
        """,
        (user_id,),
    ).fetchall()

    result: Dict[int, Dict[str, Optional[float]]] = {}
    for incident_id, score_sum, sample_count in rows:
        sample_count_value = int(sample_count)
        success_rate = (
            round(float(score_sum) / sample_count_value, 3)
            if sample_count_value > 0
            else None
        )
        result[int(incident_id)] = {
            "historical_success_rate": success_rate,
            "outcome_samples": sample_count_value,
        }
    return result


async def embed_text(text: str) -> Optional[List[float]]:
    """
    Calls Gemini's embedding endpoint and returns the resulting vector.

    Returns None on failure rather than raising - embedding is a
    "nice-to-have" enhancement layered on top of the core analysis flow,
    so a failure here should degrade gracefully (skip similarity
    matching for this request) rather than take down the whole
    /api/analyze call. The caller decides what "no embedding" means for
    the request.
    """
    try:
        # embed_content is synchronous in the current SDK; if this ever
        # becomes a real bottleneck under load, wrap it with
        # asyncio.to_thread(...) to avoid blocking the event loop. Left
        # as a direct call here since a single embedding call is fast
        # (well under a second) relative to the main generation call.
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except GoogleAPICallError as e:
        logger.warning(f"Embedding call failed (API error): {e}")
        return None
    except Exception as e:
        logger.warning(f"Embedding call failed (unexpected): {e}")
        return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Standard cosine similarity: dot(a, b) / (|a| * |b|).
    Measures the angle between two vectors, not their magnitude - which
    is what you want for embeddings, since two texts about the "same
    topic" should point in a similar direction regardless of length.
    """
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def store_incident(
    user_id: int,
    error_log: str,
    environment: str,
    analysis_text: str,
    embedding: List[float],
) -> int:
    """
    Persists a newly-analyzed incident so future requests can match
    against it. Called AFTER a successful analysis completes - we only
    want to store incidents that actually produced a real result, not
    partial/failed attempts.

    `user_id` ties the incident to whoever ran the analysis - this is
    what makes per-user dashboards and per-user similarity matching
    possible. Returns the new incident's id, which the Day 2 dashboard
    endpoints will use to fetch a single incident's full detail.

    Severity is extracted from analysis_text ONCE, here, at write time,
    rather than re-parsed with a regex every time the dashboard list is
    read - incidents are read far more often than written, so this is
    the cheaper place to do the work. `status` starts at its DB default
    ('Open') for every new incident.
    """
    severity = extract_severity(analysis_text)
    fingerprint = build_incident_fingerprint(error_log)
    conn = sqlite3.connect(DB_PATH)
    try:
        duplicate_row = conn.execute(
            """
            SELECT id
            FROM incidents
            WHERE user_id = ? AND fingerprint = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (user_id, fingerprint),
        ).fetchone()
        duplicate_of_id = duplicate_row[0] if duplicate_row else None

        cursor = conn.execute(
            """
            INSERT INTO incidents (
                user_id, error_log, environment, analysis_text, embedding, severity, fingerprint, duplicate_of_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                error_log,
                environment,
                analysis_text,
                json.dumps(embedding),
                severity,
                fingerprint,
                duplicate_of_id,
            ),
        )
        new_incident_id = cursor.lastrowid
        _insert_event(
            conn=conn,
            incident_id=new_incident_id,
            user_id=user_id,
            event_type="incident_created",
            new_status="Open",
            note=f"Incident analyzed in {environment}.",
        )
        if duplicate_of_id:
            _insert_event(
                conn=conn,
                incident_id=new_incident_id,
                user_id=user_id,
                event_type="incident_deduplicated",
                note=f"Fingerprint match with prior incident #{duplicate_of_id}.",
            )
        conn.commit()
        return new_incident_id
    finally:
        conn.close()


def find_similar(
    query_embedding: List[float], user_id: int, exclude_id: Optional[int] = None
) -> List[SimilarIncident]:
    """
    Brute-force scans this user's stored incidents, computes cosine
    similarity against the query embedding, and returns the top matches
    above SIMILARITY_THRESHOLD, sorted best-first.

    Scoped to `user_id` deliberately - in a multi-user tool, "similar past
    incidents" should mean the current user's own history, not a shared
    pool of every user's data. This also avoids leaking one user's error
    logs/analyses into another user's results.

    `exclude_id` exists so that if this function is ever called AFTER
    storing the current incident (rather than before), the incident
    doesn't trivially match itself with similarity 1.0.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, error_log, environment, analysis_text, embedding, created_at "
            "FROM incidents WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        outcome_stats = _get_outcome_stats_for_incidents(conn, user_id)
    finally:
        conn.close()

    scored: List[SimilarIncident] = []
    for row in rows:
        incident_id, error_log, environment, analysis_text, embedding_json, created_at = row
        if exclude_id is not None and incident_id == exclude_id:
            continue
        try:
            stored_embedding = json.loads(embedding_json)
        except json.JSONDecodeError:
            continue
        similarity = cosine_similarity(query_embedding, stored_embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            stats = outcome_stats.get(incident_id, {})
            historical_success_rate = stats.get("historical_success_rate")
            outcome_samples = int(stats.get("outcome_samples", 0) or 0)
            outcome_component = (
                historical_success_rate if historical_success_rate is not None else 0.5
            )
            ranking_score = (
                (similarity * SIMILARITY_WEIGHT)
                + (outcome_component * OUTCOME_WEIGHT)
            )
            scored.append(
                SimilarIncident(
                    incident_id=incident_id,
                    error_log=error_log,
                    environment=environment,
                    analysis_text=analysis_text,
                    created_at=created_at,
                    similarity=similarity,
                    historical_success_rate=historical_success_rate,
                    outcome_samples=outcome_samples,
                    ranking_score=ranking_score,
                )
            )

    scored.sort(
        key=lambda incident: (
            incident.ranking_score,
            incident.similarity,
            incident.outcome_samples,
        ),
        reverse=True,
    )
    return scored[:MAX_MATCHES]


def _build_incident_where_clause(
    user_id: int,
    environment: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    duplicate_only: bool = False,
) -> Tuple[str, List[object]]:
    # sqlite parameter list can contain mixed primitives (int/str/etc.)
    # depending on active filters.
    where_parts = ["user_id = ?"]
    params: List[object] = [user_id]

    if environment:
        where_parts.append("environment = ?")
        params.append(environment)
    if severity:
        where_parts.append("severity = ?")
        params.append(severity)
    if status:
        where_parts.append("status = ?")
        params.append(status)
    if search:
        like = f"%{search}%"
        where_parts.append("(error_log LIKE ? OR analysis_text LIKE ?)")
        params.extend([like, like])
    if duplicate_only:
        where_parts.append("duplicate_of_id IS NOT NULL")

    return " AND ".join(where_parts), params


def _seed_sample_incidents_for_user(conn: sqlite3.Connection, user_id: int) -> None:
    sample_1_log = "2026-07-22 14:03:11 [ERROR] [payment-worker] ConnectionResetError: [Errno 104] Connection reset by peer while POSTing to https://api.payment-gateway.internal:8443/v2/charges"
    sample_1_analysis = """## 1. Root Cause Analysis
[SEVERITY: Sev2]

The provided **Python** log indicates a `ConnectionResetError` while the payment worker attempted to reach the downstream payment gateway under high traffic.

- [HIGH-CONFIDENCE] Upstream TLS termination began failing during peak load.
- [HIGH-CONFIDENCE] Outbound HTTP call in `payment_worker.py` line 84 timed out without retry logic.
- [INFERRED] Network reset caused unhandled exception and pod restart loop.

## 2. Architecture Impact
```mermaid
flowchart LR
    A[Payment Worker] -->|HTTPS :8443| B[Payment Gateway]
    B --> C[(Database)]
    style B fill:#ff4d4d,stroke:#900,color:#fff
```

## 3. Remediation / Git Patch
```diff
--- a/app/services/payment_worker.py
+++ b/app/services/payment_worker.py
@@ -84,7 +84,10 @@ def process_payment(payload):
-    response = requests.post(PAYMENT_GATEWAY_URL, json=payload, timeout=5)
+    session = requests.Session()
+    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
+    session.mount("https://", HTTPAdapter(max_retries=retries))
+    response = session.post(PAYMENT_GATEWAY_URL, json=payload, timeout=10)
```
"""

    sample_2_log = "2026-07-22 11:20:05 [CRITICAL] [postgresql-db] FATAL: remaining connection slots are reserved for non-replication superuser connections. Max connections 100 reached."
    sample_2_analysis = """## 1. Root Cause Analysis
[SEVERITY: Sev3]

PostgreSQL database connection pool reached the maximum limit of 100 active connections in **PostgreSQL** environment.

- [HIGH-CONFIDENCE] Leaked connections from unclosed ORM sessions in background task workers.
- [INFERRED] Sudden spike in background job processing created transient connection exhaustion.

## 2. Architecture Impact
```mermaid
flowchart TD
    A[API Workers] -->|Connection Pool| B[(PostgreSQL DB)]
    style B fill:#eab308,stroke:#a16207,color:#fff
```

## 3. Remediation / Git Patch
```diff
--- a/app/db/session.py
+++ b/app/db/session.py
@@ -12,3 +12,3 @@
-engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
+engine = create_engine(DATABASE_URL, pool_size=25, max_overflow=50, pool_pre_ping=True, pool_recycle=1800)
```
"""

    f1 = build_incident_fingerprint(sample_1_log)
    f2 = build_incident_fingerprint(sample_2_log)

    cursor1 = conn.execute(
        """
        INSERT INTO incidents (user_id, error_log, environment, analysis_text, embedding, severity, status, fingerprint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, sample_1_log, "Python", sample_1_analysis, json.dumps([]), "Sev2", "Investigating", f1)
    )
    _insert_event(conn, cursor1.lastrowid, user_id, "incident_created", new_status="Investigating", note="Initial incident auto-populated for workspace.")

    cursor2 = conn.execute(
        """
        INSERT INTO incidents (user_id, error_log, environment, analysis_text, embedding, severity, status, fingerprint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, sample_2_log, "PostgreSQL", sample_2_analysis, json.dumps([]), "Sev3", "Open", f2)
    )
    _insert_event(conn, cursor2.lastrowid, user_id, "incident_created", new_status="Open", note="Initial incident auto-populated for workspace.")

    conn.commit()


def seed_sample_incidents_if_empty(conn: sqlite3.Connection, user_id: int) -> None:
    try:
        user_inc_count = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        if user_inc_count == 0:
            _seed_sample_incidents_for_user(conn, user_id)
    except Exception as e:
        logger.warning(f"Error checking or seeding sample incidents for user {user_id}: {e}")


def get_incidents_for_user(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
    environment: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    duplicate_only: bool = False,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> Dict[str, object]:
    """
    Returns a user's incident history, most recent first, without the
    embedding field (irrelevant to a dashboard listing and needlessly
    large to send over the wire). Used by Day 2's dashboard endpoint, and
    by today's tests to verify incidents are correctly tied to the user
    who created them.
    """
    normalized_sort_by = SORTABLE_INCIDENT_FIELDS.get(sort_by, "created_at")
    normalized_sort_order = "ASC" if sort_order.lower() == "asc" else "DESC"

    conn = sqlite3.connect(DB_PATH)
    try:
        # Seed initial realistic sample incidents if user has none
        if not (environment or severity or status or search or duplicate_only):
            seed_sample_incidents_if_empty(conn, user_id)

        where_clause, params = _build_incident_where_clause(
            user_id=user_id,
            environment=environment,
            severity=severity,
            status=status,
            search=search,
            duplicate_only=duplicate_only,
        )
        total_count_row = conn.execute(
            # where_clause is built entirely from hardcoded fragments in
            # _build_incident_where_clause() (e.g. "user_id = ?"); every
            # actual value is still bound via the `params` list below as
            # `?` placeholders, never string-interpolated.
            f"SELECT COUNT(*) FROM incidents WHERE {where_clause}",  # nosec B608
            params,
        ).fetchone()
        total_count = total_count_row[0] if total_count_row else 0

        rows = conn.execute(
            # Same where_clause guarantee as above. The two other
            # interpolated fragments here are also never raw user input:
            # normalized_sort_by is looked up through the
            # SORTABLE_INCIDENT_FIELDS allowlist dict (falls back to
            # "created_at" for anything not in it), and
            # normalized_sort_order is hardcoded to literally "ASC" or
            # "DESC" by a ternary - neither can carry attacker-controlled
            # text into the query.
            f"""
            SELECT
                i.id,
                i.error_log,
                i.environment,
                i.analysis_text,
                i.created_at,
                i.severity,
                i.status,
                i.fingerprint,
                i.duplicate_of_id,
                i.jira_issue_key,
                i.github_pr_url,
                CASE
                    WHEN i.fingerprint IS NULL THEN 1
                    ELSE (
                        SELECT COUNT(*)
                        FROM incidents i2
                        WHERE i2.user_id = i.user_id AND i2.fingerprint = i.fingerprint
                    )
                END AS duplicate_count
            FROM incidents
            AS i
            WHERE {where_clause}
            ORDER BY i.{normalized_sort_by} {normalized_sort_order}
            LIMIT ? OFFSET ?
            """,  # nosec B608
            [*params, limit, offset],
        ).fetchall()
        outcome_stats = _get_outcome_stats_for_incidents(conn, user_id)
    finally:
        conn.close()

    incidents = [
        {
            "id": row[0],
            "error_log_preview": row[1][:200],
            "environment": row[2],
            "analysis_preview": row[3][:300],
            "created_at": row[4],
            "severity": row[5],
            "status": row[6],
            "fingerprint": row[7],
            "duplicate_of_id": row[8],
            "jira_issue_key": row[9],
            "github_pr_url": row[10],
            "duplicate_count": row[11],
            "historical_success_rate": outcome_stats.get(row[0], {}).get("historical_success_rate"),
            "outcome_samples": outcome_stats.get(row[0], {}).get("outcome_samples", 0),
        }
        for row in rows
    ]
    return {"incidents": incidents, "total_count": total_count}


def get_incident_by_id(incident_id: int, user_id: int) -> Optional[dict]:
    """
    Returns the FULL stored record (including the complete analysis_text,
    not just a preview) for one incident - used when a user clicks into a
    specific past incident from their dashboard to see the full RCA,
    diagram, and diff again.

    Critically, this is scoped by BOTH incident_id AND user_id in the
    WHERE clause - not just incident_id. Without the user_id check here,
    a logged-in user could view ANY incident by guessing/incrementing IDs
    in the URL (an "insecure direct object reference" vulnerability) -
    scoping the query itself is what actually prevents that, not just
    hiding the button in the UI.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT
                i.id,
                i.error_log,
                i.environment,
                i.analysis_text,
                i.created_at,
                i.severity,
                i.status,
                i.resolved_at,
                i.fingerprint,
                i.duplicate_of_id,
                i.jira_issue_key,
                i.github_pr_url,
                i.slack_thread_ts,
                CASE
                    WHEN i.fingerprint IS NULL THEN 1
                    ELSE (
                        SELECT COUNT(*)
                        FROM incidents i2
                        WHERE i2.user_id = i.user_id AND i2.fingerprint = i.fingerprint
                    )
                END AS duplicate_count
            FROM incidents
            AS i
            WHERE i.id = ? AND i.user_id = ?
            """,
            (incident_id, user_id),
        ).fetchone()
        outcome_stats = _get_outcome_stats_for_incidents(conn, user_id).get(incident_id, {})
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "error_log": row[1],
        "environment": row[2],
        "analysis_text": row[3],
        "created_at": row[4],
        "severity": row[5],
        "status": row[6],
        "resolved_at": row[7],
        "fingerprint": row[8],
        "duplicate_of_id": row[9],
        "jira_issue_key": row[10],
        "github_pr_url": row[11],
        "slack_thread_ts": row[12],
        "duplicate_count": row[13],
        "historical_success_rate": outcome_stats.get("historical_success_rate"),
        "outcome_samples": outcome_stats.get("outcome_samples", 0),
    }


def update_incident_status(incident_id: int, user_id: int, new_status: str) -> Optional[dict]:
    """
    Updates an incident's workflow status (Open / Investigating /
    Resolved). Returns the updated {id, status, resolved_at} on success,
    or None if no matching row was found FOR THIS USER - same IDOR
    protection pattern as get_incident_by_id: the WHERE clause checks
    user_id as well as incident_id, so a user can't change the status of
    an incident that isn't theirs just by guessing/incrementing an id in
    the URL, regardless of what the frontend does or doesn't show them.

    `resolved_at` is set to the current time when moving TO "Resolved",
    and cleared back to NULL if moved away from "Resolved" (e.g. someone
    reopens it) - so it always reflects the most recent resolution, not
    a stale timestamp from a prior resolve/reopen cycle.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}.")

    conn = sqlite3.connect(DB_PATH)
    try:
        current_row = conn.execute(
            "SELECT status FROM incidents WHERE id = ? AND user_id = ?",
            (incident_id, user_id),
        ).fetchone()
        if current_row is None:
            return None

        previous_status = current_row[0]
        resolved_at_value = "datetime('now')" if new_status == "Resolved" else "NULL"
        conn.execute(
            # resolved_at_value is one of exactly two hardcoded literals
            # chosen by the ternary above ("datetime('now')" or "NULL"),
            # never derived from new_status text itself; new_status is
            # separately validated against VALID_STATUSES before this
            # point and is still passed as a `?` parameter, not
            # interpolated.
            f"""
            UPDATE incidents
            SET status = ?, resolved_at = {resolved_at_value}
            WHERE id = ? AND user_id = ?
            """,  # nosec B608
            (new_status, incident_id, user_id),
        )

        row = conn.execute(
            "SELECT id, status, resolved_at FROM incidents WHERE id = ? AND user_id = ?",
            (incident_id, user_id),
        ).fetchone()
        if previous_status != new_status:
            _insert_event(
                conn=conn,
                incident_id=incident_id,
                user_id=user_id,
                event_type="status_changed",
                previous_status=previous_status,
                new_status=new_status,
                note=f"Status moved from {previous_status} to {new_status}.",
            )
        conn.commit()
    finally:
        conn.close()

    return {"id": row[0], "status": row[1], "resolved_at": row[2]}


def set_incident_jira_key(incident_id: int, user_id: int, jira_issue_key: str) -> Optional[dict]:
    """
    Records the Jira issue key created/linked for this incident, so a
    later click on "Create Jira Ticket" knows to add a comment to the
    existing ticket instead of opening a duplicate one. Scoped by
    user_id for the same IDOR reasons as every other incident mutation
    here.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "UPDATE incidents SET jira_issue_key = ? WHERE id = ? AND user_id = ?",
            (jira_issue_key, incident_id, user_id),
        )
        if cursor.rowcount == 0:
            return None
        _insert_event(
            conn=conn,
            incident_id=incident_id,
            user_id=user_id,
            event_type="jira_ticket_linked",
            note=f"Linked Jira issue {jira_issue_key}.",
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": incident_id, "jira_issue_key": jira_issue_key}


def set_incident_github_pr(incident_id: int, user_id: int, pr_url: str) -> Optional[dict]:
    """Records the GitHub PR URL scaffolded for this incident's patch."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "UPDATE incidents SET github_pr_url = ? WHERE id = ? AND user_id = ?",
            (pr_url, incident_id, user_id),
        )
        if cursor.rowcount == 0:
            return None
        _insert_event(
            conn=conn,
            incident_id=incident_id,
            user_id=user_id,
            event_type="github_pr_created",
            note=f"Opened GitHub PR scaffold: {pr_url}.",
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": incident_id, "github_pr_url": pr_url}


def set_incident_slack_thread(incident_id: int, user_id: int, thread_ts: str) -> Optional[dict]:
    """
    Records the Slack message `ts` for the FIRST bot-mode post about this
    incident, so subsequent notifications (status changes, outcome
    feedback) can reply in-thread rather than posting a new top-level
    message each time. Only relevant when the user has configured a Slack
    bot token + channel (see auth.py); webhook-only Slack setups have no
    `ts` to thread against.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "UPDATE incidents SET slack_thread_ts = ? WHERE id = ? AND user_id = ?",
            (thread_ts, incident_id, user_id),
        )
        if cursor.rowcount == 0:
            return None
        conn.commit()
    finally:
        conn.close()
    return {"id": incident_id, "slack_thread_ts": thread_ts}


def log_incident_event(
    incident_id: int,
    user_id: int,
    event_type: str,
    note: Optional[str] = None,
) -> None:
    """Appends an event entry for this incident/user pair."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _insert_event(
            conn=conn,
            incident_id=incident_id,
            user_id=user_id,
            event_type=event_type,
            note=note,
        )
        conn.commit()
    finally:
        conn.close()


def get_incident_events(incident_id: int, user_id: int, limit: int = 100) -> List[dict]:
    """Returns newest-first incident activity events for one incident."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT id, event_type, previous_status, new_status, note, created_at
            FROM incident_events
            WHERE incident_id = ? AND user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (incident_id, user_id, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": row[0],
            "event_type": row[1],
            "previous_status": row[2],
            "new_status": row[3],
            "note": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]


def record_incident_outcome(
    incident_id: int,
    user_id: int,
    outcome: str,
    note: str = "",
) -> dict:
    """
    Persists user feedback about whether the suggested remediation worked.
    Returns updated outcome summary metrics for that incident.
    """
    normalized_outcome = outcome.strip().lower()
    if normalized_outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome '{outcome}'. Must be one of {VALID_OUTCOMES}.")

    conn = sqlite3.connect(DB_PATH)
    try:
        exists = conn.execute(
            "SELECT 1 FROM incidents WHERE id = ? AND user_id = ?",
            (incident_id, user_id),
        ).fetchone()
        if exists is None:
            return {}

        conn.execute(
            """
            INSERT INTO incident_outcomes (incident_id, user_id, outcome, note)
            VALUES (?, ?, ?, ?)
            """,
            (incident_id, user_id, normalized_outcome, note.strip() or None),
        )
        _insert_event(
            conn=conn,
            incident_id=incident_id,
            user_id=user_id,
            event_type="outcome_recorded",
            note=f"Outcome marked as '{normalized_outcome}'.",
        )

        summary_row = conn.execute(
            """
            SELECT
                COUNT(*) AS sample_count,
                SUM(CASE outcome WHEN 'worked' THEN 1.0 WHEN 'partial' THEN 0.5 ELSE 0.0 END) AS score_sum,
                SUM(CASE WHEN outcome = 'worked' THEN 1 ELSE 0 END) AS worked_count,
                SUM(CASE WHEN outcome = 'partial' THEN 1 ELSE 0 END) AS partial_count,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) AS failed_count
            FROM incident_outcomes
            WHERE incident_id = ? AND user_id = ?
            """,
            (incident_id, user_id),
        ).fetchone()
        conn.commit()
    finally:
        conn.close()

    sample_count = int(summary_row[0]) if summary_row and summary_row[0] is not None else 0
    score_sum = float(summary_row[1]) if summary_row and summary_row[1] is not None else 0.0
    return {
        "incident_id": incident_id,
        "sample_count": sample_count,
        "worked_count": int(summary_row[2] or 0),
        "partial_count": int(summary_row[3] or 0),
        "failed_count": int(summary_row[4] or 0),
        "historical_success_rate": round(score_sum / sample_count, 3) if sample_count else None,
    }


def get_incident_outcomes(incident_id: int, user_id: int, limit: int = 100) -> List[dict]:
    """Returns newest-first outcome feedback entries for a single incident."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT id, outcome, note, created_at
            FROM incident_outcomes
            WHERE incident_id = ? AND user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (incident_id, user_id, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": row[0],
            "outcome": row[1],
            "note": row[2] or "",
            "created_at": row[3],
        }
        for row in rows
    ]


def get_incident_analytics(user_id: int, days: int = ANALYTICS_DEFAULT_DAYS) -> dict:
    """
    Computes aggregate metrics for a user's incidents:
    - total/open/investigating/resolved counts
    - severity breakdown
    - top environments
    - incidents by day for the requested range
    - average MTTR in hours for resolved incidents
    """
    bounded_days = max(1, min(days, ANALYTICS_MAX_DAYS))
    conn = sqlite3.connect(DB_PATH)
    try:
        seed_sample_incidents_if_empty(conn, user_id)

        total_count = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

        status_rows = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM incidents
            WHERE user_id = ?
            GROUP BY status
            """,
            (user_id,),
        ).fetchall()
        status_counts = {row[0]: row[1] for row in status_rows}

        severity_rows = conn.execute(
            """
            SELECT severity, COUNT(*)
            FROM incidents
            WHERE user_id = ?
            GROUP BY severity
            """,
            (user_id,),
        ).fetchall()
        severity_breakdown = {
            severity: 0 for severity in VALID_SEVERITIES
        }
        for severity, count in severity_rows:
            severity_breakdown[severity if severity in severity_breakdown else "Unknown"] += count

        environment_rows = conn.execute(
            """
            SELECT environment, COUNT(*) as cnt
            FROM incidents
            WHERE user_id = ?
            GROUP BY environment
            ORDER BY cnt DESC, environment ASC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()

        daily_rows = conn.execute(
            # bounded_days is `max(1, min(days, ANALYTICS_MAX_DAYS))`, an
            # int clamped to a fixed numeric range (and `days` itself is
            # already constrained by FastAPI's `Query(..., ge=1, le=180)`
            # upstream in main.py) - it can never carry a SQL-injection
            # payload.
            f"""
            SELECT date(created_at) as day, COUNT(*)
            FROM incidents
            WHERE user_id = ? AND created_at >= datetime('now', '-{bounded_days - 1} days')
            GROUP BY day
            ORDER BY day ASC
            """,  # nosec B608
            (user_id,),
        ).fetchall()
        daily_map = {row[0]: row[1] for row in daily_rows}

        mttr_row = conn.execute(
            """
            SELECT AVG((julianday(resolved_at) - julianday(created_at)) * 24.0)
            FROM incidents
            WHERE user_id = ? AND status = 'Resolved' AND resolved_at IS NOT NULL
            """,
            (user_id,),
        ).fetchone()
        mttr_hours = round(mttr_row[0], 2) if mttr_row and mttr_row[0] is not None else None

        resolved_count = status_counts.get("Resolved", 0)
        resolution_rate = round((resolved_count / total_count) * 100, 2) if total_count else 0.0

        duplicate_count_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM incidents
            WHERE user_id = ? AND duplicate_of_id IS NOT NULL
            """,
            (user_id,),
        ).fetchone()
        duplicate_incident_count = int(duplicate_count_row[0]) if duplicate_count_row else 0

        outcome_row = conn.execute(
            """
            SELECT
                COUNT(*) AS sample_count,
                SUM(CASE WHEN outcome = 'worked' THEN 1 ELSE 0 END) AS worked_count,
                SUM(CASE WHEN outcome = 'partial' THEN 1 ELSE 0 END) AS partial_count,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                SUM(CASE outcome WHEN 'worked' THEN 1.0 WHEN 'partial' THEN 0.5 ELSE 0.0 END) AS score_sum
            FROM incident_outcomes
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    from datetime import datetime, timedelta

    start_date = datetime.utcnow().date() - timedelta(days=bounded_days - 1)
    incidents_by_day = []
    for i in range(bounded_days):
        date_value = start_date + timedelta(days=i)
        day_str = date_value.isoformat()
        incidents_by_day.append({"date": day_str, "count": daily_map.get(day_str, 0)})

    outcome_sample_count = int(outcome_row[0] or 0) if outcome_row else 0
    outcome_score_sum = float(outcome_row[4] or 0.0) if outcome_row else 0.0

    return {
        "days": bounded_days,
        "total_incidents": total_count,
        "open_incidents": status_counts.get("Open", 0),
        "investigating_incidents": status_counts.get("Investigating", 0),
        "resolved_incidents": resolved_count,
        "resolution_rate_percent": resolution_rate,
        "mean_time_to_resolve_hours": mttr_hours,
        "duplicate_incidents": duplicate_incident_count,
        "duplicate_rate_percent": round((duplicate_incident_count / total_count) * 100, 2) if total_count else 0.0,
        "severity_breakdown": severity_breakdown,
        "outcome_summary": {
            "sample_count": outcome_sample_count,
            "worked_count": int(outcome_row[1] or 0) if outcome_row else 0,
            "partial_count": int(outcome_row[2] or 0) if outcome_row else 0,
            "failed_count": int(outcome_row[3] or 0) if outcome_row else 0,
            "historical_success_rate": (
                round(outcome_score_sum / outcome_sample_count, 3)
                if outcome_sample_count
                else None
            ),
        },
        "top_environments": [
            {"environment": row[0], "count": row[1]} for row in environment_rows
        ],
        "incidents_by_day": incidents_by_day,
    }


def format_matches_for_prompt(matches: List[SimilarIncident]) -> str:
    """
    Turns retrieved matches into a compact text block to inject into the
    LLM's user message as extra context. Kept short (root cause snippet,
    not the full analysis) to avoid bloating the prompt - the model
    doesn't need the entire old Mermaid diagram/diff, just enough to
    recognize "this is the same failure pattern."
    """
    if not matches:
        return ""

    blocks = []
    for match in matches:
        # Pull just the Root Cause Analysis section from the stored
        # analysis rather than the whole thing, to keep this concise.
        snippet = match.analysis_text.split("## 2.")[0].replace("## 1. Root Cause Analysis", "").strip()
        snippet = snippet[:600]  # hard cap so one huge past incident can't dominate the prompt
        outcome_text = (
            f", historical_fix_success={match.historical_success_rate:.0%} "
            f"(n={match.outcome_samples})"
            if match.historical_success_rate is not None and match.outcome_samples > 0
            else ""
        )
        blocks.append(
            f"- Similarity {match.similarity:.0%}, from {match.created_at}, "
            f"environment={match.environment}{outcome_text}:\n  {snippet}"
        )
    return "\n\n".join(blocks)