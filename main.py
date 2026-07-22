"""
PatchOps - Enterprise Incident Resolution Hub - FastAPI Backend
----------------------------------------------------------------------
Two main endpoints:

  POST /api/analyze      - streams a 3-section incident analysis (Root
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

Phase E - Integrations + automation
------------------------------------
This phase wires the tool into a real incident-response workflow instead
of just producing an analysis and leaving it in the browser:

  - Slack: enriched notifications (severity color, deep link button,
    Jira/GitHub links if present). If the user has configured a Slack
    BOT token + channel (instead of, or in addition to, an Incoming
    Webhook), notifications use chat.postMessage and thread every
    follow-up event (status change, outcome feedback) under the first
    message for that incident - a plain Incoming Webhook has no `ts` to
    thread against, so that path still posts standalone messages.
  - Jira: creates a ticket from an incident's analysis on first click,
    and adds a comment to the SAME ticket on subsequent clicks (tracked
    via incidents.jira_issue_key) rather than opening duplicates.
  - GitHub: scaffolds a PR from the incident's git diff by opening a new
    branch and committing the diff as a proposed-patch file for a human
    to review and apply - this is intentionally a *scaffold*, not an
    auto-merge, since applying an LLM-authored diff unattended is not
    something this tool should ever do silently.
  - Generic outbound webhook: POSTs a JSON payload to a user-configured
    URL on incident_created / status_changed / outcome_recorded events,
    for wiring into anything not natively integrated (PagerDuty, a
    custom automation, etc). Fire-and-forget: failures are logged, never
    surfaced to the user or allowed to block the request that triggered
    them.
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
from typing import AsyncGenerator, List, Optional

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
logger = logging.getLogger("patchops")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
# Explicitly coerce MOCK_MODE so it respects false/0/off cleanly
_mock_env = os.getenv("MOCK_MODE", "false").strip().lower()
MOCK_MODE = _mock_env in ("true", "1", "yes", "on")

PORT = int(os.getenv("PORT", 8000))
CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

FREE_TIER_RPM = int(os.getenv("FREE_TIER_RPM", "10"))

APP_BASE_URL = os.getenv("APP_BASE_URL", f"http://localhost:{PORT}").rstrip("/")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

incident_store.init_db()

app = FastAPI(title="PatchOps")


# ---------------------------------------------------------------------------
# 1b. IN-MEMORY RATE LIMITER
# ---------------------------------------------------------------------------
class SlidingWindowRateLimiter:
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
    label: str
    content: str


class IncidentAnalysisRequest(BaseModel):
    error_log: str
    source_code: str = ""
    environment: str = "General"
    additional_logs: List[LogSource] = []


class PostMortemRequest(BaseModel):
    error_log: str
    analysis_text: str


class StatusUpdateRequest(BaseModel):
    status: str


class SettingsUpdateRequest(BaseModel):
    slack_webhook_url: str = ""
    slack_bot_token: str = ""
    slack_channel_id: str = ""
    jira_base_url: str = ""
    jira_project_key: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    github_repo: str = ""
    github_token: str = ""
    generic_webhook_url: str = ""
    generic_webhook_events: str = "all"


class OutcomeRecordRequest(BaseModel):
    outcome: str
    note: str = ""


class ExternalIssueRequest(BaseModel):
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
[HIGH-CONFIDENCE] pattern match. If they are NOT relevant, ignore them.

{similar_context}"""

    multi_log_instruction = ""
    if is_multi_log:
        multi_log_instruction = """

MULTI-LOG CORRELATION (required): The user has provided logs from MULTIPLE
services for the same incident, each labeled with its source. Do not
analyze each log in isolation. Instead:
  1. Identify chronological order of events ACROSS all logs using timestamps.
  2. Build a timeline as part of Section 1 using this exact format:
     [TIMELINE] <service label>: <what happened> (<timestamp if known>)
  3. State clearly which service's failure is the UPSTREAM/originating cause."""

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
3, or 4 (Sev1 - Critical, Sev2 - High, Sev3 - Medium, Sev4 - Low).

Plain-English explanation of what went wrong and why, referencing specific
lines/signals from the log(s) and code.

CONFIDENCE TAGGING (required): Every bullet point or claim in this section
must start with exactly one of these two tags:
  - `[HIGH-CONFIDENCE]` - for claims directly evidenced by the log(s) or code.
  - `[INFERRED]` - for plausible reasoning or domain knowledge.

## 2. Architecture Impact
Output a Mermaid.js **flowchart** (use `flowchart LR` or `flowchart TD`)
showing the components involved in the failure and how the error
propagates between them. **CRITICAL REQUIREMENT:** Do NOT use `sequenceDiagram` 
because it causes styling compilation errors. The diagram MUST be enclosed 
in a fenced code block tagged exactly ```mermaid and ```. The failing/originating 
component MUST be visually highlighted in red using valid flowchart style syntax:
    style ComponentName fill:#ff4d4d,stroke:#900,color:#fff
Keep node labels short. Do not include any prose inside the mermaid block itself.

## 3. Remediation / Git Patch
Give the exact code fix formatted as a valid unified git diff, enclosed in
a fenced code block tagged exactly ```diff and ```. Use standard diff
headers (--- a/path, +++ b/path, @@ hunk markers). After the diff block, briefly 
list any manual follow-up steps (e.g. restart commands) that are not part of 
the code change itself.

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
    payload = {"done": True}
    if incident_id is not None:
        payload["incident_id"] = incident_id
    return f"data: {json.dumps(payload)}\n\n"


def sse_similar_incidents(matches: list) -> str:
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
# 4b. SHARED TEXT EXTRACTION HELPERS
# ---------------------------------------------------------------------------
def extract_fenced_block(markdown: str, lang: str) -> Optional[str]:
    match = re.search(rf"```{lang}\n(.*?)```", markdown, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_clean_snippet(analysis_text: str, max_chars: int = 500) -> str:
    section_one = analysis_text.split("## 2.")[0]
    section_one = section_one.replace("## 1. Root Cause Analysis", "")
    cleaned = re.sub(r"\[SEVERITY:\s*Sev[1-4]\]", "", section_one, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[HIGH-CONFIDENCE\]\s*", "", cleaned)
    cleaned = re.sub(r"\[INFERRED\]\s*", "", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def build_incident_deep_link(incident_id: int) -> str:
    return f"{APP_BASE_URL}/?incident={incident_id}"


# ---------------------------------------------------------------------------
# 5. MOCK GENERATOR (for free frontend testing)
# ---------------------------------------------------------------------------
async def mock_analysis_stream(
    error_log: str, environment: str, user_id: int, additional_logs: List[LogSource] = None
) -> AsyncGenerator[str, None]:
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

    fake_embedding = [(hash(f"{error_log}{i}") % 1000) / 1000.0 for i in range(16)]
    new_incident_id = incident_store.store_incident(
        user_id=user_id,
        error_log=error_log,
        environment=environment,
        analysis_text=fake_response,
        embedding=fake_embedding,
    )
    dispatch_generic_webhook_background(user_id, "incident_created", new_incident_id)

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
    if not GEMINI_API_KEY:
        yield sse_error("GEMINI_API_KEY is not set on the server.")
        yield sse_done()
        return

    additional_logs = additional_logs or []
    is_multi_log = len(additional_logs) > 0

    combined_log_text = error_log
    if is_multi_log:
        combined_log_text += "\n\n" + "\n\n".join(
            f"[{log.label}]\n{log.content}" for log in additional_logs
        )

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

    user_message = f"### PRIMARY ERROR LOG\n```\n{error_log}\n```"
    for log in additional_logs:
        user_message += f"\n\n### LOG: {log.label}\n```\n{log.content}\n```"
    if source_code.strip():
        user_message += f"\n\n### SOURCE CODE\n```\n{source_code}\n```"

    if matches:
        yield sse_similar_incidents(matches)

    full_response_text = ""
    new_incident_id = None

    try:
        # Exponential backoff retry loop for handling 503 high demand spikes safely
        max_retries = 4
        delay = 2.0
        response = None

        for attempt in range(max_retries):
            try:
                response = await model.generate_content_async(
                    user_message,
                    stream=True,
                    request_options={"timeout": 60},
                )
                break
            except Exception as api_err:
                err_str = str(api_err)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower()) and attempt < max_retries - 1:
                    logger.warning(f"Gemini 503 high demand spike. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    delay *= 2.0  # Double wait time each round (2s -> 4s -> 8s)
                else:
                    raise api_err

        async for chunk in response:
            if await request.is_disconnected():
                logger.info("Client disconnected mid-stream; stopping generation.")
                break
            if chunk.text:
                full_response_text += chunk.text
                yield sse_event(chunk.text)

        if full_response_text.strip() and query_embedding:
            new_incident_id = incident_store.store_incident(
                user_id=user_id,
                error_log=combined_log_text,
                environment=environment,
                analysis_text=full_response_text,
                embedding=query_embedding,
            )
            dispatch_generic_webhook_background(user_id, "incident_created", new_incident_id)

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
SEVERITY_COLORS = {
    "Sev1": "#dc2626",
    "Sev2": "#ea580c",
    "Sev3": "#ca8a04",
    "Sev4": "#2563eb",
}
DEFAULT_SEVERITY_COLOR = "#94a3b8"


def build_slack_blocks(incident: dict) -> list:
    severity = incident.get("severity", "Unknown")
    snippet = extract_clean_snippet(incident.get("analysis_text", ""))
    deep_link = build_incident_deep_link(incident.get("id"))

    context_bits = [f"*Environment:*\n{incident.get('environment', 'General')}",
                    f"*Detected:*\n{incident.get('created_at', 'Unknown')}"]

    links_line_parts = []
    if incident.get("jira_issue_key"):
        links_line_parts.append(f"Jira: `{incident['jira_issue_key']}`")
    if incident.get("github_pr_url"):
        links_line_parts.append(f"<{incident['github_pr_url']}|GitHub PR>")

    blocks = [
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
            "fields": [{"type": "mrkdwn", "text": bit} for bit in context_bits],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": snippet or "_No summary available._"},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Full Analysis", "emoji": True},
                    "url": deep_link,
                }
            ],
        },
    ]
    if links_line_parts:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(links_line_parts)}],
        })
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Sent from PatchOps"}],
    })
    return blocks


def build_slack_payload(incident: dict) -> dict:
    severity = incident.get("severity", "Unknown")
    color = SEVERITY_COLORS.get(severity, DEFAULT_SEVERITY_COLOR)
    return {"attachments": [{"color": color, "blocks": build_slack_blocks(incident)}]}


async def send_slack_notification(webhook_url: str, incident: dict) -> tuple:
    payload = build_slack_payload(incident)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)
        if response.status_code == 200:
            return True, "Sent to Slack successfully."
        return False, f"Slack rejected the request: {response.text[:200]}"
    except httpx.TimeoutException:
        return False, "Timed out connecting to Slack. Please try again."
    except httpx.RequestError as e:
        return False, f"Could not reach Slack: {str(e)[:200]}"


async def send_slack_bot_message(
    bot_token: str, channel_id: str, blocks: list, thread_ts: Optional[str] = None
) -> tuple:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}"},
                json={
                    "channel": channel_id,
                    "blocks": blocks,
                    "thread_ts": thread_ts,
                    "text": "Incident update from PatchOps",
                },
            )
        data = response.json()
        if data.get("ok"):
            return True, "Sent to Slack successfully.", data.get("ts")
        return False, f"Slack API rejected the request: {data.get('error', 'unknown_error')}", None
    except httpx.TimeoutException:
        return False, "Timed out connecting to Slack.", None
    except httpx.RequestError as e:
        return False, f"Could not reach Slack: {str(e)[:200]}", None


async def notify_slack_for_incident(user_id: int, incident: dict, is_follow_up: bool = False) -> tuple:
    settings = auth.get_integration_settings(user_id)
    blocks = build_slack_blocks(incident)

    if settings.get("slack_bot_token") and settings.get("slack_channel_id"):
        existing_thread_ts = incident.get("slack_thread_ts") if is_follow_up else None
        success, message, ts = await send_slack_bot_message(
            settings["slack_bot_token"],
            settings["slack_channel_id"],
            blocks,
            thread_ts=existing_thread_ts,
        )
        if success and ts and not existing_thread_ts:
            incident_store.set_incident_slack_thread(incident["id"], user_id, ts)
        return success, message, False

    if settings.get("slack_webhook_url"):
        success, message = await send_slack_notification(settings["slack_webhook_url"], incident)
        return success, message, False

    return False, "Slack is not configured. Add a webhook URL or bot token in Settings first.", True


# ---------------------------------------------------------------------------
# 6c. JIRA TICKET CREATION / UPDATE
# ---------------------------------------------------------------------------
def build_jira_description_adf(incident: dict, include_diff: bool) -> dict:
    snippet = extract_clean_snippet(incident.get("analysis_text", ""), max_chars=2000)
    deep_link = build_incident_deep_link(incident.get("id"))

    content = [
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": f"Environment: {incident.get('environment', 'General')}"}],
        },
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": snippet or "No root-cause summary available."}],
        },
    ]

    if include_diff:
        diff_code = extract_fenced_block(incident.get("analysis_text", ""), "diff")
        if diff_code:
            content.append({
                "type": "codeBlock",
                "attrs": {"language": "diff"},
                "content": [{"type": "text", "text": diff_code[:4000]}],
            })

    content.append({
        "type": "paragraph",
        "content": [
            {"type": "text", "text": "Full analysis: "},
            {
                "type": "text",
                "text": deep_link,
                "marks": [{"type": "link", "attrs": {"href": deep_link}}],
            },
        ],
    })

    return {"type": "doc", "version": 1, "content": content}


async def create_or_update_jira_issue(
    user_id: int, incident: dict, title_override: str, include_diff: bool
) -> dict:
    settings = auth.get_integration_settings(user_id)
    base_url = settings.get("jira_base_url", "").rstrip("/")
    project_key = settings.get("jira_project_key", "")
    email = settings.get("jira_email", "")
    api_token = settings.get("jira_api_token", "")

    if not (base_url and project_key and email and api_token):
        raise HTTPException(
            status_code=400,
            detail="Jira is not fully configured. Add your Jira base URL, project key, email, and API token in Settings.",
        )

    auth_header = base64.b64encode(f"{email}:{api_token}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    severity = incident.get("severity", "Unknown")
    title = title_override.strip() or f"[{severity}] Incident #{incident['id']} - {incident.get('environment', 'General')}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if incident.get("jira_issue_key"):
                comment_body = build_jira_description_adf(incident, include_diff)
                response = await client.post(
                    f"{base_url}/rest/api/3/issue/{incident['jira_issue_key']}/comment",
                    headers=headers,
                    json={"body": comment_body},
                )
                if response.status_code not in (200, 201):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Jira rejected the comment update: {response.text[:300]}",
                    )
                issue_key = incident["jira_issue_key"]
            else:
                response = await client.post(
                    f"{base_url}/rest/api/3/issue",
                    headers=headers,
                    json={
                        "fields": {
                            "project": {"key": project_key},
                            "summary": title,
                            "description": build_jira_description_adf(incident, include_diff),
                            "issuetype": {"name": "Bug"},
                        }
                    },
                )
                if response.status_code not in (200, 201):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Jira rejected the ticket creation: {response.text[:300]}",
                    )
                issue_key = response.json().get("key")
                if not issue_key:
                    raise HTTPException(status_code=502, detail="Jira did not return an issue key.")
                incident_store.set_incident_jira_key(incident["id"], user_id, issue_key)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timed out connecting to Jira.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Jira: {str(e)[:200]}")

    return {"jira_issue_key": issue_key, "jira_url": f"{base_url}/browse/{issue_key}"}


# ---------------------------------------------------------------------------
# 6d. GITHUB PR SCAFFOLD
# ---------------------------------------------------------------------------
async def create_github_pr_scaffold(
    user_id: int, incident: dict, title_override: str, include_diff: bool
) -> dict:
    settings = auth.get_integration_settings(user_id)
    repo = settings.get("github_repo", "").strip().strip("/")
    token = settings.get("github_token", "")

    if not (repo and token):
        raise HTTPException(
            status_code=400,
            detail="GitHub is not configured. Add your repo (owner/name) and a token in Settings.",
        )
    if "/" not in repo:
        raise HTTPException(
            status_code=400,
            detail="GitHub repo must be in 'owner/repo' format.",
        )

    if incident.get("github_pr_url"):
        return {"pr_url": incident["github_pr_url"], "already_existed": True}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    branch_name = f"patchops/incident-{incident['id']}"
    severity = incident.get("severity", "Unknown")
    title = title_override.strip() or f"[{severity}] Proposed fix for incident #{incident['id']}"
    snippet = extract_clean_snippet(incident.get("analysis_text", ""), max_chars=2000)
    diff_code = extract_fenced_block(incident.get("analysis_text", ""), "diff") if include_diff else None
    deep_link = build_incident_deep_link(incident["id"])

    file_body_lines = [
        f"# Incident #{incident['id']} - Proposed Remediation",
        "",
        f"**Environment:** {incident.get('environment', 'General')}  ",
        f"**Severity:** {severity}  ",
        f"**Full analysis:** {deep_link}",
        "",
        "## Root Cause Summary",
        snippet or "No summary available.",
    ]
    if diff_code:
        file_body_lines += ["", "## Proposed Diff", "```diff", diff_code, "```"]
    file_body_lines += [
        "",
        "_This file was scaffolded automatically by PatchOps. Review the diff "
        "above carefully before applying it - it was generated from an LLM analysis "
        "of the incident log, not verified against this repository's current source._",
    ]
    file_content = "\n".join(file_body_lines)
    file_path = f"incident-reports/incident-{incident['id']}.md"

    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            repo_resp = await client.get(f"https://api.github.com/repos/{repo}")
            if repo_resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"GitHub rejected the repo lookup: {repo_resp.text[:300]}",
                )
            default_branch = repo_resp.json().get("default_branch", "main")

            ref_resp = await client.get(f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}")
            if ref_resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"GitHub rejected the base branch lookup: {ref_resp.text[:300]}",
                )
            base_sha = ref_resp.json()["object"]["sha"]

            create_ref_resp = await client.post(
                f"https://api.github.com/repos/{repo}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )
            if create_ref_resp.status_code not in (201, 422):
                raise HTTPException(
                    status_code=502,
                    detail=f"GitHub rejected branch creation: {create_ref_resp.text[:300]}",
                )

            put_resp = await client.put(
                f"https://api.github.com/repos/{repo}/contents/{file_path}",
                json={
                    "message": f"Add incident report + proposed patch for incident #{incident['id']}",
                    "content": base64.b64encode(file_content.encode("utf-8")).decode("ascii"),
                    "branch": branch_name,
                },
            )
            if put_resp.status_code not in (200, 201):
                raise HTTPException(
                    status_code=502,
                    detail=f"GitHub rejected the file commit: {put_resp.text[:300]}",
                )

            pr_resp = await client.post(
                f"https://api.github.com/repos/{repo}/pulls",
                json={
                    "title": title,
                    "head": branch_name,
                    "base": default_branch,
                    "body": (
                        f"Auto-scaffolded from PatchOps for incident #{incident['id']}.\n\n"
                        f"{snippet or ''}\n\nFull analysis: {deep_link}\n\n"
                        "This PR adds a reviewable incident report + proposed diff. "
                        "**The diff has not been applied to any source file in this repo** - "
                        "review it in `incident-reports/` and apply the relevant change yourself."
                    ),
                },
            )
            if pr_resp.status_code not in (200, 201):
                raise HTTPException(
                    status_code=502,
                    detail=f"GitHub rejected PR creation: {pr_resp.text[:300]}",
                )
            pr_url = pr_resp.json().get("html_url")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timed out connecting to GitHub.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach GitHub: {str(e)[:200]}")

    if not pr_url:
        raise HTTPException(status_code=502, detail="GitHub did not return a PR URL.")

    incident_store.set_incident_github_pr(incident["id"], user_id, pr_url)
    return {"pr_url": pr_url, "already_existed": False}


# ---------------------------------------------------------------------------
# 6e. GENERIC OUTBOUND WEBHOOK
# ---------------------------------------------------------------------------
async def dispatch_generic_webhook(user_id: int, event_type: str, incident_id: int) -> None:
    try:
        settings = auth.get_integration_settings(user_id)
        webhook_url = settings.get("generic_webhook_url", "")
        if not webhook_url:
            return

        subscribed_events = settings.get("generic_webhook_events", "all")
        if subscribed_events != "all":
            subscribed = {e.strip() for e in subscribed_events.split(",") if e.strip()}
            if event_type not in subscribed:
                return

        incident = incident_store.get_incident_by_id(incident_id=incident_id, user_id=user_id)
        if incident is None:
            return

        payload = {
            "event": event_type,
            "incident_id": incident_id,
            "severity": incident.get("severity"),
            "status": incident.get("status"),
            "environment": incident.get("environment"),
            "created_at": incident.get("created_at"),
            "deep_link": build_incident_deep_link(incident_id),
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code >= 300:
                logger.warning(
                    f"Generic webhook for user {user_id} returned {resp.status_code} "
                    f"for event {event_type} on incident {incident_id}."
                )
    except Exception as e:
        logger.warning(f"Generic webhook dispatch failed (non-fatal): {e}")


def dispatch_generic_webhook_background(user_id: int, event_type: str, incident_id: int) -> None:
    try:
        asyncio.create_task(dispatch_generic_webhook(user_id, event_type, incident_id))
    except RuntimeError:
        logger.warning("Could not schedule generic webhook dispatch - no running event loop.")


# ---------------------------------------------------------------------------
# 7. ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/register", response_model=auth.TokenResponse)
async def register(payload: auth.RegisterRequest):
    username = payload.username.strip()
    if not username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    try:
        user_id = auth.create_user(username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token = auth.create_access_token(user_id, username)
    logger.info(f"New user registered: {username} (id={user_id})")
    return auth.TokenResponse(access_token=token, username=username)


@app.post("/api/login", response_model=auth.TokenResponse)
async def login(payload: auth.LoginRequest):
    user = auth.authenticate_user(payload.username.strip(), payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = auth.create_access_token(user["id"], user["username"])
    return auth.TokenResponse(access_token=token, username=user["username"])


@app.get("/api/me")
async def get_me(current_user: dict = Depends(auth.get_current_user)):
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
    limit = max(1, min(limit, 100))
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
    return incident_store.get_incident_analytics(
        user_id=current_user["user_id"], days=days
    )


@app.get("/api/incidents/export")
async def export_incidents_csv(current_user: dict = Depends(auth.get_current_user)):
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
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return incident


@app.get("/api/incidents/{incident_id}/events")
async def list_incident_events(
    incident_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: dict = Depends(auth.get_current_user),
):
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

    dispatch_generic_webhook_background(current_user["user_id"], "outcome_recorded", incident_id)

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

    dispatch_generic_webhook_background(current_user["user_id"], "status_changed", incident_id)

    return result


@app.get("/api/settings")
async def get_settings(current_user: dict = Depends(auth.get_current_user)):
    return auth.get_integration_settings_for_api(current_user["user_id"])


@app.put("/api/settings")
async def update_settings(
    payload: SettingsUpdateRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    for url_field in ("slack_webhook_url", "jira_base_url", "generic_webhook_url"):
        value = getattr(payload, url_field).strip()
        if value and not value.startswith("https://"):
            raise HTTPException(
                status_code=400,
                detail=f"{url_field} must start with https:// (or leave blank to clear it).",
            )

    github_repo = payload.github_repo.strip()
    if github_repo and "/" not in github_repo:
        raise HTTPException(
            status_code=400,
            detail="github_repo must be in 'owner/repo' format (or leave blank to clear it).",
        )

    events = payload.generic_webhook_events.strip() or "all"
    if events != "all":
        invalid = [
            e.strip() for e in events.split(",")
            if e.strip() and e.strip() not in incident_store.WEBHOOK_EVENT_TYPES
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown webhook event type(s): {', '.join(invalid)}. "
                       f"Valid types: {', '.join(incident_store.WEBHOOK_EVENT_TYPES)}, or 'all'.",
            )

    updates = {
        "slack_webhook_url": payload.slack_webhook_url,
        "slack_channel_id": payload.slack_channel_id,
        "jira_base_url": payload.jira_base_url,
        "jira_project_key": payload.jira_project_key,
        "jira_email": payload.jira_email,
        "github_repo": payload.github_repo,
        "generic_webhook_url": payload.generic_webhook_url,
        "generic_webhook_events": events,
    }
    for secret_field, secret_value in (
        ("slack_bot_token", payload.slack_bot_token),
        ("jira_api_token", payload.jira_api_token),
        ("github_token", payload.github_token),
    ):
        if secret_value.strip():
            updates[secret_field] = secret_value

    auth.update_integration_settings(current_user["user_id"], updates)
    return auth.get_integration_settings_for_api(current_user["user_id"])


@app.post("/api/incidents/{incident_id}/notify-slack")
async def notify_slack(
    incident_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    success, message, not_configured = await notify_slack_for_incident(
        current_user["user_id"], incident, is_follow_up=bool(incident.get("slack_thread_ts"))
    )
    if not success:
        raise HTTPException(status_code=400 if not_configured else 502, detail=message)

    incident_store.log_incident_event(
        incident_id=incident_id,
        user_id=current_user["user_id"],
        event_type="slack_notification_sent",
        note="Incident summary sent to Slack.",
    )
    return {"status": "sent", "message": message}


@app.post("/api/incidents/{incident_id}/jira")
async def create_jira_ticket(
    incident_id: int,
    payload: ExternalIssueRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    result = await create_or_update_jira_issue(
        current_user["user_id"], incident, payload.title, payload.include_diff
    )
    return result


@app.post("/api/incidents/{incident_id}/github-pr")
async def create_github_pr(
    incident_id: int,
    payload: ExternalIssueRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    incident = incident_store.get_incident_by_id(
        incident_id=incident_id, user_id=current_user["user_id"]
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    result = await create_github_pr_scaffold(
        current_user["user_id"], incident, payload.title, payload.include_diff
    )
    return result


@app.post("/api/analyze")
async def analyze_incident(
    payload: IncidentAnalysisRequest,
    request: Request,
    current_user: dict = Depends(auth.get_current_user),
):
    if not payload.error_log.strip():
        raise HTTPException(status_code=400, detail="error_log cannot be empty.")

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
    if not FRONTEND_TEMPLATE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend template not found at templates/index.html.",
        )
    return FileResponse(str(FRONTEND_TEMPLATE_PATH))


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "model": GEMINI_MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "rate_limit_rpm": FREE_TIER_RPM,
        "embedding_model": incident_store.EMBEDDING_MODEL,
        "similarity_threshold": incident_store.SIMILARITY_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)