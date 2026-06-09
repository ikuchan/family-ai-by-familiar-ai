"""Tests for RelationshipTracker (Phase 6 — relationship modeling).

Tracks first session, session count, conversation count, and surfaces
context for the system prompt.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from familiar_agent.relationship import RelationshipTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker() -> RelationshipTracker:
    return RelationshipTracker()


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------


def test_fresh_tracker_has_zero_sessions() -> None:
    t = _tracker()
    assert t.session_count == 0


def test_fresh_tracker_has_zero_conversations() -> None:
    t = _tracker()
    assert t.conversation_count == 0


def test_fresh_tracker_has_no_first_session_date() -> None:
    t = _tracker()
    assert t.first_session_date is None


# ---------------------------------------------------------------------------
# Tests: record_session
# ---------------------------------------------------------------------------


def test_record_session_increments_count() -> None:
    t = _tracker()
    t.record_session()
    assert t.session_count == 1


def test_record_session_twice_increments_to_two() -> None:
    t = _tracker()
    t.record_session()
    t.record_session()
    assert t.session_count == 2


def test_record_session_sets_first_session_date_on_first_call() -> None:
    t = _tracker()
    t.record_session()
    assert t.first_session_date is not None


def test_record_session_does_not_overwrite_first_session_date() -> None:
    t = _tracker()
    t.record_session()
    first = t.first_session_date
    t.record_session()
    assert t.first_session_date == first


def test_record_session_updates_last_session_date() -> None:
    t = _tracker()
    t.record_session()
    assert t.last_session_date is not None


# ---------------------------------------------------------------------------
# Tests: record_conversation
# ---------------------------------------------------------------------------


def test_record_conversation_increments_count() -> None:
    t = _tracker()
    t.record_conversation()
    assert t.conversation_count == 1


def test_record_conversation_multiple_times() -> None:
    t = _tracker()
    for _ in range(5):
        t.record_conversation()
    assert t.conversation_count == 5


# ---------------------------------------------------------------------------
# Tests: days_together
# ---------------------------------------------------------------------------


def test_days_together_none_when_no_first_session() -> None:
    t = _tracker()
    assert t.days_together is None


def test_days_together_zero_on_first_day() -> None:
    t = _tracker()
    t.record_session()
    assert t.days_together == 0


def test_days_together_correct_after_backdating() -> None:
    t = _tracker()
    t.record_session()
    t._state["first_session_date"] = (date.today() - timedelta(days=7)).isoformat()
    t._save()
    assert t.days_together == 7


# ---------------------------------------------------------------------------
# Tests: persistence across instances
# ---------------------------------------------------------------------------


def test_state_persists_across_instances() -> None:
    t1 = _tracker()
    t1.record_session()
    t1.record_conversation()
    t1.record_conversation()

    t2 = _tracker()
    assert t2.session_count == 1
    assert t2.conversation_count == 2


# ---------------------------------------------------------------------------
# Tests: context_for_prompt
# ---------------------------------------------------------------------------


def test_context_for_prompt_returns_string() -> None:
    t = _tracker()
    t.record_session()
    t.record_conversation()
    result = t.context_for_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_context_for_prompt_empty_before_first_session() -> None:
    t = _tracker()
    result = t.context_for_prompt()
    assert result == "" or result is None


def test_context_for_prompt_includes_session_info() -> None:
    t = _tracker()
    t.record_session()
    for _ in range(3):
        t.record_conversation()
    result = t.context_for_prompt()
    assert (
        "session" in result.lower() or "conversation" in result.lower() or "talk" in result.lower()
    )


def test_relationship_tracks_trust_and_intimacy_trajectory() -> None:
    t = _tracker()
    t.note_trust_shift(0.7, "shared a vulnerable moment")
    t.note_intimacy_shift(0.66, "celebrated together")

    assert t.trust == 0.7
    assert t.intimacy == 0.66


def test_relationship_records_support_preferences_and_permissions() -> None:
    t = _tracker()
    t.record_support_preference("validate first before advice")
    t.set_permission("offer_unsolicited_advice", False, evidence="asked to slow down")

    ctx = t.relational_context_for_prompt()
    assert "support-preferences" in ctx
    assert "permissions-blocked" in ctx


def test_relationship_imports_legacy_json_into_db(tmp_path: Path) -> None:
    legacy_path = tmp_path / "relationship.json"
    legacy_path.write_text(
        """
        {
          "first_session_date": "2026-04-01",
          "session_count": 2,
          "conversation_count": 4,
          "trust_trajectory": [{"value": 0.81, "evidence": "legacy"}]
        }
        """.strip(),
        encoding="utf-8",
    )

    tracker = RelationshipTracker(state_path=legacy_path)
    assert tracker.session_count == 2

    reloaded = RelationshipTracker()
    assert reloaded.trust == 0.81
