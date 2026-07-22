"""
Unit tests for incident_store.py's pure logic - fingerprinting, cosine
similarity, severity extraction, and outcome-aware ranking. These don't
need the FastAPI app at all, just an isolated DB where storage is involved.
"""

import math

import pytest

import incident_store
from tests.conftest import create_incident_via_store


# ---------------------------------------------------------------------------
# extract_severity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("[SEVERITY: Sev1]\nSome text", "Sev1"),
        ("blah blah [SEVERITY: Sev4] more text", "Sev4"),
        ("[severity: sev2]", "Sev2"),  # case-insensitive
        ("no severity tag here at all", "Unknown"),
        ("[SEVERITY: Sev9]", "Unknown"),  # out of range, regex won't match
    ],
)
def test_extract_severity(text, expected):
    assert incident_store.extract_severity(text) == expected


# ---------------------------------------------------------------------------
# build_incident_fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_ignores_volatile_values():
    """
    Two logs that differ only in timestamp/numeric/hex noise should
    fingerprint identically - that's the whole point of normalizing
    those out before hashing (dedup should catch "same error, different
    request id / timestamp" as duplicates).
    """
    log_a = "2026-07-08 14:22:10 ERROR conn reset at 0xDEADBEEF, retry=3"
    log_b = "2026-07-09 03:11:58 ERROR conn reset at 0xCAFEBABE, retry=7"
    assert incident_store.build_incident_fingerprint(log_a) == incident_store.build_incident_fingerprint(log_b)


def test_fingerprint_differs_for_different_errors():
    log_a = "ConnectionResetError while calling payment gateway"
    log_b = "NullPointerException in inventory service"
    assert incident_store.build_incident_fingerprint(log_a) != incident_store.build_incident_fingerprint(log_b)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------
def test_cosine_similarity_identical_vectors_is_one():
    v = [0.1, 0.2, 0.3, 0.4]
    assert math.isclose(incident_store.cosine_similarity(v, v), 1.0, rel_tol=1e-9)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert incident_store.cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_similarity_mismatched_length_returns_zero():
    assert incident_store.cosine_similarity([1, 2, 3], [1, 2]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero():
    assert incident_store.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------------------
# store_incident + find_similar (requires isolated_db fixture)
# ---------------------------------------------------------------------------
def test_store_and_find_similar_above_threshold(isolated_db):
    user_id = 1
    base_embedding = [1.0, 0.0, 0.0, 0.0]
    incident_store.store_incident(
        user_id=user_id,
        error_log="ConnectionResetError at line 84",
        environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- [HIGH-CONFIDENCE] x",
        embedding=base_embedding,
    )

    # Near-identical embedding -> should be found as similar.
    matches = incident_store.find_similar([0.99, 0.05, 0.0, 0.0], user_id=user_id)
    assert len(matches) == 1
    assert matches[0].similarity >= incident_store.SIMILARITY_THRESHOLD


def test_find_similar_excludes_dissimilar_vectors(isolated_db):
    user_id = 1
    incident_store.store_incident(
        user_id=user_id,
        error_log="ConnectionResetError",
        environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    # Orthogonal query embedding -> similarity 0, well below threshold.
    matches = incident_store.find_similar([0.0, 1.0, 0.0, 0.0], user_id=user_id)
    assert matches == []


def test_find_similar_scoped_to_user(isolated_db):
    """
    Incidents belonging to a different user must never surface in
    another user's similarity results - this is the core privacy
    guarantee of the RAG retrieval layer.
    """
    incident_store.store_incident(
        user_id=1,
        error_log="ConnectionResetError",
        environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    matches = incident_store.find_similar([1.0, 0.0, 0.0, 0.0], user_id=2)
    assert matches == []


def test_store_incident_marks_duplicate_by_fingerprint(isolated_db):
    user_id = 1
    same_log = "ConnectionResetError: retry attempt 1 at 2026-07-08 14:22:10"
    first_id = incident_store.store_incident(
        user_id=user_id, error_log=same_log, environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x", embedding=[0.1] * 8,
    )
    # Same underlying error, different volatile details -> same fingerprint.
    second_log = "ConnectionResetError: retry attempt 9 at 2026-07-09 03:11:58"
    second_id = incident_store.store_incident(
        user_id=user_id, error_log=second_log, environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x", embedding=[0.1] * 8,
    )

    detail = incident_store.get_incident_by_id(second_id, user_id)
    assert detail["duplicate_of_id"] == first_id


# ---------------------------------------------------------------------------
# Outcome-aware ranking
# ---------------------------------------------------------------------------
def test_ranking_score_prefers_higher_historical_success(isolated_db):
    user_id = 1
    query_embedding = [1.0, 0.0, 0.0, 0.0]

    # Two incidents with IDENTICAL similarity to the query, but different
    # recorded outcomes - the one with a better track record should rank
    # higher, since ranking_score = similarity*WEIGHT + outcome*WEIGHT.
    good_id = incident_store.store_incident(
        user_id=user_id, error_log="err A", environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x", embedding=[1.0, 0.0, 0.0, 0.0],
    )
    bad_id = incident_store.store_incident(
        user_id=user_id, error_log="err B", environment="Docker",
        analysis_text="## 1. Root Cause Analysis\n[SEVERITY: Sev2]\n- x", embedding=[1.0, 0.0, 0.0, 0.0],
    )
    incident_store.record_incident_outcome(good_id, user_id, "worked")
    incident_store.record_incident_outcome(bad_id, user_id, "failed")

    matches = incident_store.find_similar(query_embedding, user_id=user_id)
    assert len(matches) == 2
    assert matches[0].incident_id == good_id  # higher-ranked comes first
    assert matches[0].ranking_score > matches[1].ranking_score


def test_record_incident_outcome_rejects_invalid_value(isolated_db):
    incident_id = create_incident_via_store(user_id=1)
    with pytest.raises(ValueError):
        incident_store.record_incident_outcome(incident_id, 1, "sort-of-worked")


def test_update_incident_status_rejects_invalid_value(isolated_db):
    incident_id = create_incident_via_store(user_id=1)
    with pytest.raises(ValueError):
        incident_store.update_incident_status(incident_id, 1, "Escalated")


def test_update_incident_status_scoped_to_owner(isolated_db):
    """A user can't update the status of an incident they don't own."""
    incident_id = create_incident_via_store(user_id=1)
    result = incident_store.update_incident_status(incident_id, user_id=2, new_status="Resolved")
    assert result is None