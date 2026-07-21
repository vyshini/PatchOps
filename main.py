"""
SRE AI Copilot - Enterprise Incident Resolution Hub - FastAPI Backend
----------------------------------------------------------------------
Two main endpoints:

  POST /api/analyze     - streams a 3-section incident analysis (Root
                           Cause Analysis with confidence tagging and
                           optional multi-log correlation timeline,
                           Architecture Impact as a Mermaid.js diagram,
                           Remediation as a git diff) back to the client
                           via Server-Sent Events (SSE). Also retrieves
                           and injects semantically similar past
                           incidents (RAG) before generating.

  POST /api/postmortem   - non-streaming; takes the original error_log
                           plus the analysis_text already produced by
                           /api/analyze, and returns a formal Incident
                           Post-Mortem document as raw markdown for the
                           frontend to download as a .md file.

Mock and real generators for /api/analyze produce the same "shape" of
output (an async stream of text chunks), so the endpoint and the frontend
don't need to know or care whether MOCK_MODE is on or off.
"""

import os
import json
import asyncio
import logging
import time
import csv
import io
import re
import base64
from pathlib import Path
from collections import deque
from typing import AsyncGenerator, List

import httpx
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    DeadlineExceeded,
    InvalidArgument,
    GoogleAPICallError,
)
from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import incident_store
import auth

# ---------------------------------------------------------------------------
# 1. CONFIG & STARTUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sre-copilot")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

FREE_TIER_RPM = int(os.getenv("FREE_TIER_RPM", "10"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initializes the SQLite incidents table used for the "similar past
# incidents" retrieval feature. Safe to call on every startup - it's a
# CREATE TABLE IF NOT EXISTS under the hood.
incident_store.init_db()

app = FastAPI(title="SRE AI Copilot")


# ---------------------------------------------------------------------------
# 1b. IN-MEMORY RATE LIMITER
# ---------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """
    Tracks request timestamps in a rolling 60-second window and rejects
    new requests once the window is full. Protects the free Gemini quota
    from being exhausted by a burst of traffic or a buggy retry loop.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self.timestamps and now - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_requests:
            return False
        self.timestamps.append(now)
        return True


rate_limiter = SlidingWindowRateLimiter(max_requests=FREE_TIER_RPM)

if CORS_ALLOW_ORIGINS.strip() == "*":
    allowed_origins = ["*"]
    logger.warning("CORS_ALLOW_ORIGINS is '*'. This is not recommended for production.")
else:
    allowed_origins = [
        origin.strip()
        for origin in CORS_ALLOW_ORIGINS.split(",")
        if origin.strip()
    ]
    if not allowed_origins:
        raise ValueError(
            "CORS_ALLOW_ORIGINS did not contain any valid origins. "
            "Set a comma-separated allowlist or '*'."
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# 2. REQUEST SCHEMAS
# ---------------------------------------------------------------------------
class LogSource(BaseModel):
    """
    One labeled log input. `label` identifies which service/system this
    log came from (e.g. "API Gateway", "Payment Worker", "Database") so
    the model can reason about WHICH system said what when correlating
    events across multiple logs - without a label, the model has no way
    to distinguish "the error appeared in service A three seconds after
    service B logged a timeout" from noise.
    """
    label: str
    content: str


class IncidentAnalysisRequest(BaseModel):
    """
    Pydantic model = automatic request validation.
    If the client POSTs JSON that doesn't match this shape, FastAPI rejects
    it with a 422 before our code even runs.

    `source_code` is optional - a user may only have the error log and not
    the relevant file, in which case we still analyze the log alone but
    can't produce a precise git diff (the prompt accounts for this below).

    Backward-compatible request shape: `error_log` remains supported as
    the single-log path (existing frontend behavior, and the simplest
    possible request). `additional_logs` is new and optional - when
    present, the backend treats this as a multi-log correlation request
    and asks the model to build a cross-service timeline instead of
    analyzing one log in isolation.
    """
    error_log: str
    source_code: str = ""
    environment: str = "General"  # e.g. "Docker", "Python", "Kubernetes", "AWS"
    additional_logs: List[LogSource] = []


class PostMortemRequest(BaseModel):
    """
    Input for the post-mortem generator. `analysis_text` is the full RCA
    markdown that was already streamed to the client from /api/analyze -
    we reuse it as context instead of re-deriving the root cause from
    scratch, so the post-mortem stays consistent with what the user
    already saw and (presumably) acted on.
    """
    error_log: str
    analysis_text: str


class StatusUpdateRequest(BaseModel):
    """Body for PATCH /api/incidents/{id}/status. `status` is validated
    against incident_store.VALID_STATUSES inside the endpoint itself
    (not here via a Literal type) so the error message can name the
    exact valid options in a clean 400, rather than FastAPI's default
    422 validation error format."""
    status: str


class SettingsUpdateRequest(BaseModel):
    """Body for PUT /api/settings integration configuration."""
    slack_webhook_url: str = ""
    jira_base_url: str = ""
    jira_project_key: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    github_repo: str = ""
    github_token: str = ""


class OutcomeRecordRequest(BaseModel):
    """Body for POST /api/incidents/{id}/outcome."""
    outcome: str
    note: str = ""


class ExternalIssueRequest(BaseModel):
    """Optional overrides for external ticket/issue creation."""
    title: str = ""
    include_diff: bool = True


# ---------------------------------------------------------------------------
# 3. SYSTEM PROMPTS
# ---------------------------------------------------------------------------
def build_analysis_system_prompt(
    environment: str,
    has_source_code: bool,
    similar_context: str = "",
    is_multi_log: bool = False,
) -> str:
    """
    Builds the persona + strict output contract for the incident analysis
    stream. Composes three optional instruction blocks on top of the base
    prompt depending on what context is available for this request:
      - similar_context: retrieved past incidents (RAG)
      - is_multi_log: whether multiple labeled logs were provided
    """
    source_code_instruction = (
        "The user has provided relevant source code below the error log. "
        "Use it to produce an exact, applicable git diff in Section 3."
        if has_source_code
        else "The user did NOT provide source code. In Section 3, still "
        "produce a best-effort unified diff against a plausible file based "
        "on the stack trace's file paths and line numbers, but clearly "
        "note in the diff's context (as a comment) that it is inferred, "
        "not applied against real source."
    )

    similar_incident_instruction = ""
    if similar_context:
        similar_incident_instruction = f"""

PRIOR SIMILAR INCIDENTS: The system has retrieved past incidents that are
semantically similar to this one, shown below. If they are genuinely
relevant to this failure, reference them explicitly in Section 1 as a
[HIGH-CONFIDENCE] pattern match (e.g. "[HIGH-CONFIDENCE] This matches a
prior incident with {{similarity}}% similarity, where the root cause
was..."). If, after reviewing them, they are NOT actually relevant to
this specific error, ignore them entirely and do not mention them - do
not force a connection that isn't real.

{similar_context}"""

    multi_log_instruction = ""
    if is_multi_log:
        multi_log_instruction = """

MULTI-LOG CORRELATION (required): The user has provided logs from MULTIPLE
services for the same incident, each labeled with its source. Do not
analyze each log in isolation. Instead:
  1. Identify the chronological order of relevant events ACROSS all
     provided logs, using timestamps where available (or, if timestamps
     are missing/inconsistent, reasonable causal ordering).
  2. Build a short timeline as part of Section 1, using this exact format
     for each entry:
     [TIMELINE] <service label>: <what happened> (<timestamp if known>)
  3. Explicitly state which service's failure appears to be the
     UPSTREAM/originating cause and which are downstream symptoms/effects
     of it - correlation across services is the whole point, not a
     restatement of each log separately.
  4. Apply the same [HIGH-CONFIDENCE] / [INFERRED] tagging rules to
     timeline entries and causal claims as to everything else in
     Section 1."""

    return f"""You are a Senior Site Reliability Engineer (SRE) and Principal
Software Architect with 15+ years of experience in distributed systems,
{environment} environments, and incident response. A developer has pasted
a raw error log (and possibly source code) below. {source_code_instruction}{similar_incident_instruction}{multi_log_instruction}

Respond ONLY in the following strict Markdown format, with these exact
three headings, in this order, and nothing else before or after them:

## 1. Root Cause Analysis
SEVERITY CLASSIFICATION (required, must be the very first line of this
section, before any other text): classify this incident's severity using
EXACTLY this format on its own line: `[SEVERITY: SevX]` where X is 1, 2,
3, or 4. Use these criteria:
  - Sev1 - Critical: complete outage, data loss/corruption, or a security
    breach; affects all or most users with no workaround.
  - Sev2 - High: major functionality broken or badly degraded for a
    significant subset of users; no reasonable workaround exists.
  - Sev3 - Medium: partial degradation, a workaround exists, or impact is
    limited to a non-critical feature or small subset of users.
  - Sev4 - Low: minor/cosmetic issue, edge case, or negligible user
    impact.
Base this on the actual evidence in the log/code, not worst-case
speculation - an error that self-recovers or affects one edge case is
NOT automatically Sev1 just because exceptions look alarming.

Plain-English explanation of what went wrong and why, referencing specific
lines/signals from the log(s) and code.

CONFIDENCE TAGGING (required): Every bullet point or claim in this section
must start with exactly one of these two tags:
  - `[HIGH-CONFIDENCE]` - only for claims directly evidenced by the log(s)
    or source code (e.g. an exact exception type, a stack trace line, a
    status code that literally appears in the input), OR a genuine match
    against a prior similar incident provided above.
  - `[INFERRED]` - for claims that are plausible reasoning or domain
    knowledge but NOT directly visible in the provided input.
Do not skip tagging any claim. Format each as a markdown bullet, e.g.:
  - [HIGH-CONFIDENCE] The log shows a `ConnectionResetError` at line 84.
  - [INFERRED] This is likely due to the downstream gateway's TLS session
    timing out under load, though the log does not confirm the exact cause.

## 2. Architecture Impact
Output a Mermaid.js flowchart or sequence diagram (whichever fits better)
showing the components involved in the failure and how the error
propagates between them. If multiple logs were provided, the diagram MUST
reflect the cross-service flow identified in the timeline above (prefer a
sequence diagram to show ordering between services). The diagram MUST be
enclosed in a fenced code block tagged exactly ```mermaid and ```. The
failing/originating component MUST be visually highlighted in red using
Mermaid style syntax, for example:
    style ComponentName fill:#ff4d4d,stroke:#900,color:#fff
Keep node labels short. Do not include any prose inside the mermaid block
itself - all explanation goes in the surrounding markdown text.

## 3. Remediation / Git Patch
Give the exact code fix formatted as a valid unified git diff, enclosed in
a fenced code block tagged exactly ```diff and ```. Use standard diff
headers (--- a/path, +++ b/path, @@ hunk markers) so it can be rendered by
diff2html without modification. After the diff block, briefly list any
manual follow-up steps (e.g. restart commands) that are not part of the
code change itself.

Be precise and technical. Do not add any preamble before
"## 1. Root Cause Analysis" and do not add anything after Section 3."""


def build_postmortem_system_prompt() -> str:
    return """You are a Senior SRE writing an official Incident Post-Mortem
document for internal stakeholders (engineering leadership and the
on-call team). You will be given the original error log and the root
cause analysis that was already produced for this incident.

Respond ONLY in the following strict Markdown format, with these exact
headings, in this order:

## Executive Summary
2-3 sentences a non-technical stakeholder could understand: what broke,
for how long (estimate qualitatively if not stated), and the resolution
status.

## Impact
Bullet points covering affected systems/services, likely user-facing
impact, and severity (Sev1-Sev4, your best judgment). If the prior
analysis contains a multi-service timeline (marked with [TIMELINE] tags),
summarize which services were affected and in what order as part of this
section - do not discard that cross-service context.

## Root Cause
A tightened, formal restatement of the root cause (do not just repeat the
analysis verbatim - synthesize it into 1-2 precise paragraphs). Where the
prior analysis distinguished directly-evidenced facts from inferred
reasoning, preserve that distinction here explicitly - e.g. "Log evidence
confirms X; the team's working hypothesis, not yet confirmed, is Y." Do
not present inferred causes as established fact. If multiple services
were involved, state clearly which was the originating/upstream cause
and which were downstream symptoms.

## Action Items
A markdown table with columns: Action | Owner | Priority | Due By. Owner
should be a role placeholder like "Backend Team" if not specified. Include
at least one immediate action and one longer-term preventative action.

Be formal, precise, and concise - this document may be read by leadership.
Do not add any preamble or content outside these four headings."""


# ---------------------------------------------------------------------------
# 4. SSE FRAMING HELPER
# ---------------------------------------------------------------------------
def sse_event(data: str) -> str:
    return f"data: {json.dumps({'text': data})}\n\n"


def sse_error(message: str) -> str:
    return f"data: {json.dumps({'error': message})}\n\n"


def sse_done(incident_id: int = None) -> str:
    """
    Sentinel event marking stream completion. Optionally carries the id
    of the incident that was just stored, so the frontend can offer a
    "Send to Slack" action immediately, without a separate round trip to
    look up "what did I just create". None when nothing was stored (e.g.
    GEMINI_API_KEY missing, or the model produced no usable output).
    """
    payload = {"done": True}
    if incident_id is not None:
        payload["incident_id"] = incident_id
    return f"data: {json.dumps(payload)}\n\n"


def sse_similar_incidents(matches: list) -> str:
    """
    Formats retrieved similar-incident matches as their own SSE event type
    (distinct from `text`/`error`/`done`), sent BEFORE the analysis text
    starts streaming, so the frontend can render the "Similar Past
    Incidents" panel immediately rather than waiting for the full
    analysis to complete.
    """
    payload = [
        {
            "similarity": round(m.similarity, 3),
            "ranking_score": round(m.ranking_score, 3),
            "historical_success_rate": (
                round(m.historical_success_rate, 3)
                if m.historical_success_rate is not None
                else None
            ),
            "outcome_samples": m.outcome_samples,
            "environment": m.environment,
            "created_at": m.created_at,
            "error_log_preview": m.error_log[:200],
        }
        for m in matches
    ]
    return f"data: {json.dumps({'similar_incidents': payload})}\n\n"


# ---------------------------------------------------------------------------
# 5. MOCK GENERATOR (for free frontend testing)
# ---------------------------------------------------------------------------
async def mock_analysis_stream(
    error_log: str, environment: str, user_id: int, additional_logs: List[LogSource] = None
) -> AsyncGenerator[str, None]:
    """
    Yields a hardcoded incident analysis word by word, with a small delay
    between chunks, simulating the "typewriter" feel of a real LLM stream
    without calling any API or spending any tokens. Demonstrates all three
    features (confidence tagging, similar-incident matches, and multi-log
    correlation when additional_logs is non-empty) so the frontend can be
    fully built and tested against mock mode alone.
    """
    additional_logs = additional_logs or []
    is_multi_log = len(additional_logs) > 0

    fake_matches = [
        {
            "similarity": 0.91,
            "ranking_score": 0.928,
            "historical_success_rate": 0.8,
            "outcome_samples": 5,
            "environment": environment,
            "created_at": "2026-07-08 14:22:10",
            "error_log_preview": (
                "ConnectionResetError: [Errno 104] Connection reset by peer "
                "while POSTing to payment gateway..."
            ),
        }
    ]
    yield f"data: {json.dumps({'similar_incidents': fake_matches})}\n\n"
    await asyncio.sleep(0.3)

    if is_multi_log:
        fake_response = f"""## 1. Root Cause Analysis
[SEVERITY: Sev2]

Correlating **{len(additional_logs) + 1} logs** across services for this **{environment}** incident. This is a mock response - no LLM was called.

- [HIGH-CONFIDENCE] [TIMELINE] Payment Gateway: upstream TLS termination began failing (14:03:11)
- [HIGH-CONFIDENCE] [TIMELINE] Payment Worker: ConnectionResetError raised while calling gateway (14:03:14)
- [HIGH-CONFIDENCE] This matches a prior incident with 91% similarity, where the same connection-reset pattern occurred.
- [INFERRED] The Payment Gateway's failure is the originating cause; the Payment Worker's error is a downstream symptom, not the root fault.

## 2. Architecture Impact
```mermaid
sequenceDiagram
    participant G as Payment Gateway
    participant W as Payment Worker
    G-->>W: TLS termination failure
    W->>G: HTTPS request
    G--xW: Connection reset
    style G fill:#ff4d4d,stroke:#900,color:#fff
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
Restart the worker deployment after merging: `kubectl rollout restart deployment/payment-worker`
"""
    else:
        fake_response = f"""## 1. Root Cause Analysis
[SEVERITY: Sev3]

The provided **{environment}** log indicates a `ConnectionResetError` while the
payment worker attempted to reach the downstream payment gateway. This is a
mock response - no LLM was called.

- [HIGH-CONFIDENCE] The log shows a `ConnectionResetError` raised at line 84 of `payment_worker.py` during an outbound HTTPS call.
- [HIGH-CONFIDENCE] This matches a prior incident with 91% similarity, where the same connection-reset pattern occurred.
- [INFERRED] This is likely caused by a transient TLS handshake failure at the gateway under load, though the log does not explicitly confirm the gateway-side cause.

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
Restart the worker deployment after merging: `kubectl rollout restart deployment/payment-worker`
"""

    for word in fake_response.split(" "):
        yield sse_event(word + " ")
        await asyncio.sleep(0.02)

    # Store this mock incident too, using a cheap deterministic
    # "embedding" derived from the log text itself (NOT a real semantic
    # embedding - just enough structure for cosine similarity to behave
    # sensibly in a demo). This matters because store_incident() is what
    # populates BOTH the dashboard's incident history AND the similar-
    # incidents matching - without this, mock mode (which most local
    # testing and demoing uses, specifically to avoid API costs) would
    # leave the dashboard permanently empty, which would look like a bug
    # rather than the cost-saving tradeoff it actually is.
    fake_embedding = [(hash(f"{error_log}{i}") % 1000) / 1000.0 for i in range(16)]
    new_incident_id = incident_store.store_incident(
        user_id=user_id,
        error_log=error_log,
        environment=environment,
        analysis_text=fake_response,
        embedding=fake_embedding,
    )

    # BUG FIX (preserved from original): this generator must always send a
    # `done` sentinel, or the frontend's stream-completion logic never
    # fires in mock mode.
    yield sse_done(incident_id=new_incident_id)


# ---------------------------------------------------------------------------
# 6. REAL LLM GENERATOR
# ---------------------------------------------------------------------------
async def real_analysis_stream(
    error_log: str,
    source_code: str,
    environment: str,
    request: Request,
    user_id: int,
    additional_logs: List[LogSource] = None,
) -> AsyncGenerator[str, None]:
    """
    Calls the Gemini API in streaming mode and re-yields each text chunk
    as it arrives, wrapped in SSE framing.

    Full lifecycle per request:
      1. Combine all provided logs into one text blob for embedding.
      2. Embed it and search for similar past incidents (RAG retrieval).
         Non-fatal on failure - degrades to "no similar incidents" rather
         than blocking the core analysis.
      3. Build the system prompt with similar-incident context and
         multi-log correlation instructions injected as needed.
      4. Send similar-incident matches to the frontend as their own SSE
         event, BEFORE the analysis text starts streaming.
      5. Stream the analysis itself, checking for client disconnects.
      6. On successful completion, store this incident (with its already-
         computed embedding) for future retrieval.

    Every failure mode (rate limit, timeout, bad input, network) is caught
    individually and turned into a specific, actionable message - never a
    silent hang or a raw traceback leaking to the client.
    """
    if not GEMINI_API_KEY:
        yield sse_error("GEMINI_API_KEY is not set on the server.")
        yield sse_done()
        return

    additional_logs = additional_logs or []
    is_multi_log = len(additional_logs) > 0

    # Combine all logs (primary + additional) into one text blob, labeled
    # by source, for both embedding/retrieval and storage. This keeps a
    # multi-log incident searchable as a whole in future similarity
    # lookups, rather than only the primary log being represented.
    combined_log_text = error_log
    if is_multi_log:
        combined_log_text += "\n\n" + "\n\n".join(
            f"[{log.label}]\n{log.content}" for log in additional_logs
        )

    # --- Retrieval step (RAG) ---
    query_embedding = await incident_store.embed_text(combined_log_text)
    similar_context = ""
    matches: list = []
    if query_embedding:
        matches = incident_store.find_similar(query_embedding, user_id=user_id)
        similar_context = incident_store.format_matches_for_prompt(matches)
        if matches:
            logger.info(f"Found {len(matches)} similar past incident(s) above threshold.")
    else:
        logger.info("Embedding unavailable for this request; skipping similarity retrieval.")

    system_prompt = build_analysis_system_prompt(
        environment,
        has_source_code=bool(source_code.strip()),
        similar_context=similar_context,
        is_multi_log=is_multi_log,
    )
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )

    # Build the user message with clearly labeled sections per log source -
    # unambiguous labeling is what makes cross-service correlation possible
    # at all; an unlabeled concatenation would give the model no way to
    # attribute which event came from which service.
    user_message = f"### PRIMARY ERROR LOG\n```\n{error_log}\n```"
    for log in additional_logs:
        user_message += f"\n\n### LOG: {log.label}\n```\n{log.content}\n```"
    if source_code.strip():
        user_message += f"\n\n### SOURCE CODE\n```\n{source_code}\n```"

    if matches:
        yield sse_similar_incidents(matches)

    full_response_text = ""  # accumulated so we can store it after streaming completes
    new_incident_id = None  # set only if storage succeeds; carried into the final done event

    try:
        response = await model.generate_content_async(
            user_message,
            stream=True,
            request_options={"timeout": 60},  # hard server-side timeout
        )

        async for chunk in response:
            if await request.is_disconnected():
                logger.info("Client disconnected mid-stream; stopping generation.")
                break
            if chunk.text:
                full_response_text += chunk.text
                yield sse_event(chunk.text)

        # --- Storage step ---
        # Only store if we got a real response and have an embedding for
        # it. Reuses query_embedding rather than re-embedding, since the
        # log text hasn't changed since we computed it above.
        if full_response_text.strip() and query_embedding:
            new_incident_id = incident_store.store_incident(
                user_id=user_id,
                error_log=combined_log_text,
                environment=environment,
                analysis_text=full_response_text,
                embedding=query_embedding,
            )

    except ResourceExhausted:
        logger.warning("Gemini quota exhausted (ResourceExhausted).")
        yield sse_error(
            "The free Gemini quota has been hit for this minute. Please wait ~60s and retry."
        )

    except DeadlineExceeded:
        yield sse_error("The request to Gemini timed out. Try a shorter log/code excerpt.")

    except InvalidArgument as e:
        logger.warning(f"Gemini rejected the request: {e}")
        yield sse_error("The input is too long or malformed for the model. Please trim it.")

    except GoogleAPICallError as e:
        logger.error(f"Gemini API error: {e}")
        yield sse_error(f"AI provider error: {str(e)}")

    except Exception as e:
        logger.exception("Unexpected error in real_analysis_stream")
        yield sse_error(f"Unexpected server error: {str(e)}")

    finally:
        yield sse_done(incident_id=new_incident_id)


# ---------------------------------------------------------------------------
# 6b. SLACK NOTIFICATIONS
# ---------------------------------------------------------------------------
# Maps severity to a hex color for Slack's attachment "color bar" - the
# same palette as the frontend's severity badges, so the visual language
# stays consistent whether someone's looking at the dashboard or Slack.
SEVERITY_COLORS = {
    "Sev1": "#dc2626",
    "Sev2": "#ea580c",
    "Sev3": "#ca8a04",
    "Sev4": "#2563eb",
}
DEFAULT_SEVERITY_COLOR = "#94a3b8"


def extract_clean_snippet(analysis_text: str, max_chars: int = 500) -> str:
    """
    Pulls a clean, Slack-readable snippet out of Section 1 (Root Cause
    Analysis), stripping the [SEVERITY: SevX], [HIGH-CONFIDENCE], and
    [INFERRED] tags that are meant for the web UI's colored badges but
    would just read as noisy bracketed text in a plain Slack message.
    """
    section_one = analysis_text.split("## 2.")[0]
    section_one = section_one.replace("## 1. Root Cause Analysis", "")
    cleaned = re.sub(r"\[SEVERITY:\s*Sev[1-4]\]", "", section_one, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[HIGH-CONFIDENCE\]\s*", "", cleaned)
    cleaned = re.sub(r"\[INFERRED\]\s*", "", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def build_slack_payload(incident: dict) -> dict:
    """
    Builds a Slack Block Kit message (using the legacy "attachments"
    color-bar feature, which is still fully supported for exactly this
    kind of at-a-glance severity signal) for one incident.
    """
    severity = incident.get("severity", "Unknown")
    color = SEVERITY_COLORS.get(severity, DEFAULT_SEVERITY_COLOR)
    snippet = extract_clean_snippet(incident.get("analysis_text", ""))

    return {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🛠️ Incident Alert - {severity}",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Environment:*\n{incident.get('environment', 'General')}"},
                            {"type": "mrkdwn", "text": f"*Detected:*\n{incident.get('created_at', 'Unknown')}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": snippet or "_No summary available._"},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "Sent from SRE AI Copilot"}
                        ],
                    },
                ],
            }
        ]
    }


async def send_slack_notification(webhook_url: str, incident: dict) -> tuple:
    """
    Posts the incident summary to the given Slack Incoming Webhook URL.
    Returns (success: bool, message: str) rather than raising, so the
    calling endpoint can turn a failure into a clean 4xx/5xx response
    with an actionable message instead of a raw traceback - a webhook
    delivery failure is an expected, recoverable event (wrong URL,
    revoked webhook, Slack outage), not a bug in this server.
    """
    payload = build_slack_payload(incident)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)
        if response.status_code == 200:
            return True, "Sent to Slack successfully."
        # Slack's Incoming Webhooks return a descriptive plain-text body
        # on failure (e.g. "invalid_payload", "channel_not_found",
        # "no_service") - surfacing it directly is more actionable than a
        # generic "failed" message.
        return False, f"Slack rejected the request: {response.text[:200]}"
    except httpx.TimeoutException:
        return False, "Timed out connecting to Slack. Please try again."
    except httpx.RequestError as e:
        return False, f"Could not reach Slack: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# 7. ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/register", response_model=auth.TokenResponse)
async def register(payload: auth.RegisterRequest):
    """
    Creates a new user account and immediately returns a valid access
    token - registering logs you in, rather than requiring a separate
    login call right after, which would be an extra round trip for no
    real benefit in a single-tenant portfolio app like this one.
    """
    username = payload.username.strip()
    if not username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    try:
        user_id = auth.create_user(username, payload.password)
    except ValueError as e:
        # Raised by auth.create_user on a duplicate username (the
        # sqlite3.IntegrityError is caught there and turned into a
        # clean message) - surfaced here as a 409 Conflict, not a 500.
        raise HTTPException(status_code=409, detail=str(e))

    token = auth.create_access_token(user_id, username)
    logger.info(f"New user registered: {username} (id={user_id})")
    return auth.TokenResponse(access_token=token, username=username)


@app.post("/api/login", response_model=auth.TokenResponse)
async def login(payload: auth.LoginRequest):
    """
    Validates credentials and issues a fresh access token. Deliberately
    returns the SAME error message for both "username doesn't exist" and
    "password is wrong" - distinguishing the two in the response would
    let an attacker enumerate valid usernames one guess at a time.
    """
    user = auth.authenticate_user(payload.username.strip(), payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = auth.create_access_token(user["id"], user["username"])
    return auth.TokenResponse(access_token=token, username=user["username"])


@app.get("/api/me")
async def get_me(current_user: dict = Depends(auth.get_current_user)):
    """
    Lets the frontend verify a stored token is still valid on page load
    (e.g. after a refresh) and fetch the current username to display,
    without needing a bespoke "whoami" payload defined elsewhere.
    """
    return current_user


@app.get("/api/incidents")
async def list_incidents(
    limit: int = 50,
    offset: int = 0,
    environment: str = Query(default=""),
    severity: str = Query(default=""),
    status: str = Query(default=""),
    search: str = Query(default=""),
    duplicate_only: bool = Query(default=False),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Returns the current user's incident history, most recent first, as
    lightweight previews (not full analysis text - keeps the dashboard's
    initial load fast). Scoped to current_user["user_id"] only; there is
    no way to pass another user's id here since it comes from the
    validated JWT, not a request parameter.
    """
    limit = max(1, min(limit, 100))  # guardrail against absurd/abusive values
    offset = max(0, offset)

    severity_filter = None
    if severity.strip():
        severity_lookup = {item.lower(): item for item in incident_store.VALID_SEVERITIES}
        severity_filter = severity_lookup.get(severity.strip().lower())
        if severity_filter is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity. Must be one of: {', '.join(incident_store.VALID_SEVERITIES)}.",
            )

    status_filter = None
    if status.strip():
        status_lookup = {item.lower(): item for item in incident_store.VALID_STATUSES}
        status_filter = status_lookup.get(status.strip().lower())
        if status_filter is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(incident_store.VALID_STATUSES)}.",
            )

    normalized_sort_by = sort_by.strip().lower()
    if normalized_sort_by not in incident_store.SORTABLE_INCIDENT_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by. Must be one of: {', '.join(incident_store.SORTABLE_INCIDENT_FIELDS.keys())}.",
        )
    normalized_sort_order = sort_order.strip().lower()
    if normalized_sort_order not in {"asc", "desc"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid sort_order. Must be 'asc' or 'desc'.",
        )

    result = incident_store.get_incidents_for_user(
        user_id=current_user["user_id"],
        limit=limit,
        offset=offset,
        environment=environment.strip() or None,
        severity=severity_filter,
        status=status_filter,
        search=search.strip() or None,
        duplicate_only=duplicate_only,
        sort_by=normalized_sort_by,
        sort_order=normalized_sort_order,
    )
    incidents = result["incidents"]
    total_count = result["total_count"]
    return {
        "incidents": incidents,
        "count": len(incidents),
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/incidents/analytics")
async def incident_analytics(
    days: int = Query(default=30, ge=1, le=180),
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Returns aggregate incident metrics for dashboards/reporting without
    requiring the client to load and compute over every incident row.
    """
    return incident_store.get_incident_analytics(
        user_id=current_user["user_id"], days=days
    )


@app.get("/api/incidents/export")
async def export_incidents_csv(current_user: dict = Depends(auth.get_current_user)):
    """
    Exports the current user's full incident history as a downloadable
    CSV file. Uses Python's built-in csv module writing into an in-memory
    StringIO buffer, then wraps that in a StreamingResponse with a
    Content-Disposition header - the standard way to make a browser treat
    a response as a file download rather than displaying it inline.

    IMPORTANT - route ordering: this MUST be registered before
    GET /api/incidents/{incident_id} below. FastAPI/Starlette match
    routes in registration order, and {incident_id} is a wildcard path
    parameter that will happily "match" the literal string "export" and
    try to parse it as an int - which is exactly the bug this ordering
    avoids (it surfaced as a confusing 422 "unable to parse export as an
    integer" the first time this was tested here).
    """
    incidents = incident_store.get_incidents_for_user(
        user_id=current_user["user_id"], limit=10_000, offset=0
    )["incidents"]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["ID", "Environment", "Severity", "Status", "Created At", "Error Log Preview"])
    for inc in incidents:
        writer.writerow([
            inc["id"], inc["environment"], inc["severity"], inc["status"],
            inc["created_at"], inc["error_log_preview"],
        ])
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=incident_history.csv"},
    )


@app.get("/api/incidents/{incident_id}")
async def get_incident_detail(
    incident_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Returns one incident's full stored analysis (complete RCA markdown,
    not a preview) so the dashboard can re-render the original mermaid
    diagram, git diff, and confidence-tagged analysis exactly as it was
    first generated - reusing the same rendering code the analyzer view
    already has, rather than duplicating it.
    """
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        # Deliberately the same 404 whether the incident doesn't exist AT
        # ALL, or exists but belongs to someone else - distinguishing the
        # two would confirm to an attacker that a given ID is valid,
        # just owned by another user.
        raise HTTPException(status_code=404, detail="Incident not found.")
    return incident


@app.get("/api/incidents/{incident_id}/events")
async def list_incident_events(
    incident_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Returns newest-first audit trail events for one incident, scoped to
    the authenticated user.
    """
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    events = incident_store.get_incident_events(
        incident_id=incident_id, user_id=current_user["user_id"], limit=limit
    )
    return {"incident_id": incident_id, "events": events, "count": len(events)}


@app.post("/api/incidents/{incident_id}/outcome")
async def record_incident_outcome(
    incident_id: int,
    payload: OutcomeRecordRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Stores user feedback on whether the suggested remediation worked.
    This powers outcome-aware ranking for future similar incidents.
    """
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    try:
        summary = incident_store.record_incident_outcome(
            incident_id=incident_id,
            user_id=current_user["user_id"],
            outcome=payload.outcome,
            note=payload.note,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{str(e)}",
        )
    return {
        "incident_id": incident_id,
        "recorded_outcome": payload.outcome.strip().lower(),
        "summary": summary,
    }


@app.get("/api/incidents/{incident_id}/outcomes")
async def list_incident_outcomes(
    incident_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: dict = Depends(auth.get_current_user),
):
    """Returns newest-first outcome feedback entries for this incident."""
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    outcomes = incident_store.get_incident_outcomes(
        incident_id=incident_id, user_id=current_user["user_id"], limit=limit
    )
    return {"incident_id": incident_id, "outcomes": outcomes, "count": len(outcomes)}


@app.patch("/api/incidents/{incident_id}/status")
async def update_incident_status(
    incident_id: int,
    payload: StatusUpdateRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Updates an incident's workflow status (Open / Investigating /
    Resolved). Same ownership scoping as GET /api/incidents/{id}: the
    underlying query checks user_id as well as incident_id, so this
    can't be used to modify another user's incident by guessing an id.
    """
    if payload.status not in incident_store.VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(incident_store.VALID_STATUSES)}.",
        )

    result = incident_store.update_incident_status(
        incident_id=incident_id,
        user_id=current_user["user_id"],
        new_status=payload.status,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return result


@app.get("/api/settings")
async def get_settings(current_user: dict = Depends(auth.get_current_user)):
    """Returns the current user's saved integration settings (currently
    just the Slack webhook URL). Returns an empty string, not an error,
    when nothing has been configured yet - "not configured" is a normal
    state, not a failure."""
    webhook_url = auth.get_webhook_url(current_user["user_id"])
    return {"slack_webhook_url": webhook_url or ""}


@app.put("/api/settings")
async def update_settings(
    payload: SettingsUpdateRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Saves the user's Slack webhook URL. Only a loose format check is
    applied (must be empty, or a plausible https:// URL) rather than
    strictly validating it's an actual Slack hooks.slack.com domain -
    Slack occasionally changes its webhook domain conventions, and a
    false-positive rejection here is worse than letting an invalid URL
    fail later with a clear error on the actual "Send to Slack" attempt.
    """
    url = payload.slack_webhook_url.strip()
    if url and not url.startswith("https://"):
        raise HTTPException(
            status_code=400,
            detail="Webhook URL must start with https:// (or leave blank to clear it).",
        )
    auth.set_webhook_url(current_user["user_id"], url)
    return {"slack_webhook_url": url}


@app.post("/api/incidents/{incident_id}/notify-slack")
async def notify_slack(
    incident_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Sends one incident's summary to the user's configured Slack webhook.
    Two distinct failure modes are surfaced separately: no webhook
    configured (400 - user needs to visit Settings) vs. Slack itself
    rejecting/failing the delivery (502 - a downstream provider issue,
    not something wrong with this server).
    """
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    webhook_url = auth.get_webhook_url(current_user["user_id"])
    if not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="No Slack webhook configured. Add one in Settings first.",
        )

    success, message = await send_slack_notification(webhook_url, incident)
    if not success:
        raise HTTPException(status_code=502, detail=message)
    incident_store.log_incident_event(
        incident_id=incident_id,
        user_id=current_user["user_id"],
        event_type="slack_notification_sent",
        note="Incident summary sent to Slack webhook.",
    )
    return {"status": "sent", "message": message}


@app.post("/api/analyze")
async def analyze_incident(
    payload: IncidentAnalysisRequest,
    request: Request,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Entry point hit by the frontend's fetch() call for live incident
    analysis (RCA + Mermaid diagram + git diff, streamed). Supports both
    single-log (original) and multi-log (additional_logs populated)
    requests via the same schema.

    Now requires authentication: incidents are tied to the requesting
    user both for storage and for scoping "similar past incidents"
    retrieval to that user's own history, rather than a shared pool.
    """
    if not payload.error_log.strip():
        raise HTTPException(status_code=400, detail="error_log cannot be empty.")

    # Guardrail: Gemini's free tier has a token-per-request ceiling. Now
    # accounts for ALL additional labeled logs too, not just the primary
    # log and source code, since a multi-log request can get large fast.
    MAX_TOTAL_CHARS = 80_000
    additional_chars = sum(len(log.content) for log in payload.additional_logs)
    total_chars = len(payload.error_log) + len(payload.source_code) + additional_chars
    if total_chars > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Combined logs + source_code exceeds {MAX_TOTAL_CHARS} characters; please trim it.",
        )

    if not MOCK_MODE and not rate_limiter.allow():
        logger.warning("Rate limit exceeded for /api/analyze.")
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached for the free tier. Please wait a minute and try again.",
        )

    generator = (
        mock_analysis_stream(payload.error_log, payload.environment, current_user["user_id"], payload.additional_logs)
        if MOCK_MODE
        else real_analysis_stream(
            payload.error_log,
            payload.source_code,
            payload.environment,
            request,
            current_user["user_id"],
            payload.additional_logs,
        )
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/postmortem")
async def generate_postmortem(
    payload: PostMortemRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Generates a formal Incident Post-Mortem document from the original
    error log plus the analysis text already produced by /api/analyze.
    Deliberately non-streaming - see original docstring reasoning.

    Requires authentication (same as /api/analyze) - primarily to prevent
    anonymous users from draining the shared Gemini free-tier quota, since
    this endpoint doesn't otherwise need to know which user is asking.
    """
    if not payload.error_log.strip() or not payload.analysis_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Both error_log and analysis_text are required.",
        )

    if MOCK_MODE or not GEMINI_API_KEY:
        return {
            "markdown": (
                "## Executive Summary\n"
                "This is a mock post-mortem. The payment worker experienced repeated "
                "connection resets to the downstream payment gateway, causing pod "
                "restarts. Service was degraded but recovered automatically.\n\n"
                "## Impact\n"
                "- **Affected service:** payment-worker\n"
                "- **User impact:** Delayed payment processing for ~5 minutes\n"
                "- **Severity:** Sev3\n\n"
                "## Root Cause\n"
                "The payment worker's HTTP client had no retry/backoff logic and no "
                "connection pooling, so a transient network blip at the gateway's TLS "
                "layer surfaced as an unhandled exception that crashed the process.\n\n"
                "## Action Items\n"
                "| Action | Owner | Priority | Due By |\n"
                "|---|---|---|---|\n"
                "| Deploy retry/backoff patch | Backend Team | High | This mock has no real due date |\n"
                "| Add alert on connection reset rate | SRE Team | Medium | This mock has no real due date |\n"
            )
        }

    if not rate_limiter.allow():
        logger.warning("Rate limit exceeded for /api/postmortem.")
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached for the free tier. Please wait a minute and try again.",
        )

    system_prompt = build_postmortem_system_prompt()
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    user_message = (
        f"### ORIGINAL ERROR LOG\n```\n{payload.error_log}\n```\n\n"
        f"### PRIOR ANALYSIS\n{payload.analysis_text}"
    )

    try:
        response = await model.generate_content_async(
            user_message,
            request_options={"timeout": 60},
        )
        return {"markdown": response.text}

    except ResourceExhausted:
        logger.warning("Gemini quota exhausted on /api/postmortem.")
        raise HTTPException(
            status_code=429,
            detail="The free Gemini quota has been hit for this minute. Please wait ~60s and retry.",
        )

    except DeadlineExceeded:
        raise HTTPException(
            status_code=504,
            detail="The request to Gemini timed out generating the post-mortem.",
        )

    except InvalidArgument as e:
        logger.warning(f"Gemini rejected the postmortem request: {e}")
        raise HTTPException(
            status_code=400,
            detail="The input is too long or malformed for the model. Please trim it.",
        )

    except GoogleAPICallError as e:
        logger.error(f"Gemini API error on /api/postmortem: {e}")
        raise HTTPException(status_code=502, detail=f"AI provider error: {str(e)}")

    except Exception as e:
        logger.exception("Unexpected error in /api/postmortem")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@app.get("/")
async def serve_frontend():
    """Serves the single-file frontend at the root URL."""
    if not FRONTEND_TEMPLATE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend template not found at templates/index.html.",
        )
    return FileResponse(str(FRONTEND_TEMPLATE_PATH))


@app.get("/api/health")
async def health_check():
    """Liveness probe, and a quick way to confirm which mode (mock vs
    real) the deployed instance is in."""
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "model": GEMINI_MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "rate_limit_rpm": FREE_TIER_RPM,
        "embedding_model": incident_store.EMBEDDING_MODEL,
        "similarity_threshold": incident_store.SIMILARITY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 8. LOCAL DEV ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)