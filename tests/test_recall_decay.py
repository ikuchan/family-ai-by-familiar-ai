"""Tests for Issue B: time-decay scoring and recall reinforcement.

recall() uses: final_score = cosine × time_score × importance
time_score   = max(FLOOR, exp(-elapsed_days / tau))
tau          = half_life × 2^recall_count / ln(2)

recall_mode:
  "conversation" → recall_count += 1 AND last_recalled_at = now()
  "spontaneous"  → last_recalled_at = now() only
  "system"       → no reinforcement
"""

from __future__ import annotations

import os
from unittest.mock import patch

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory():
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


def _fresh_conn():
    url = os.environ.get(
        "DATABASE_URL",
        os.environ["DATABASE_URL"],
    )
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


def test_recall_count_column_exists():
    """observations.recall_count must exist with DEFAULT 0 after migration 017."""
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_default, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'observations' AND column_name = 'recall_count'
            """)
            row = cur.fetchone()
        assert row is not None, "recall_count column not found"
        assert "0" in str(row["column_default"]), f"expected DEFAULT 0, got: {row['column_default']}"
    finally:
        conn.close()


def test_last_recalled_at_column_exists():
    """observations.last_recalled_at must exist and be nullable after migration 017."""
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'observations' AND column_name = 'last_recalled_at'
            """)
            row = cur.fetchone()
        assert row is not None, "last_recalled_at column not found"
        assert row["is_nullable"] == "YES", "last_recalled_at must be nullable"
        assert "timestamp" in row["data_type"], f"unexpected type: {row['data_type']}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# recall_mode reinforcement
# ---------------------------------------------------------------------------


def test_recall_mode_param_accepted(memory):
    """recall() accepts recall_mode parameter without error."""
    memory.save("テスト記憶", kind="observation", emotion="neutral")
    result = memory.recall("テスト", n=3, recall_mode="conversation")
    assert isinstance(result, list)


def test_recall_conversation_increments_recall_count(memory):
    """recall_mode='conversation' increments recall_count on returned memories."""
    memory.save("会話想起テスト", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, recall_count FROM observations WHERE content = %s AND person_id = %s",
                ("会話想起テスト", memory._person_id),
            )
            before = cur.fetchone()
        assert before["recall_count"] == 0

        memory.recall("会話想起テスト", n=5, recall_mode="conversation")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT recall_count FROM observations WHERE id = %s",
                (before["id"],),
            )
            after = cur.fetchone()
        assert after["recall_count"] == 1, (
            f"recall_count should be 1 after conversation recall, got {after['recall_count']}"
        )
    finally:
        conn.close()


def test_recall_spontaneous_updates_last_recalled_at_only(memory):
    """recall_mode='spontaneous' updates last_recalled_at but NOT recall_count."""
    memory.save("自発想起テスト", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, recall_count, last_recalled_at FROM observations "
                "WHERE content = %s AND person_id = %s",
                ("自発想起テスト", memory._person_id),
            )
            before = cur.fetchone()
        assert before["recall_count"] == 0
        assert before["last_recalled_at"] is None

        memory.recall("自発想起テスト", n=5, recall_mode="spontaneous")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT recall_count, last_recalled_at FROM observations WHERE id = %s",
                (before["id"],),
            )
            after = cur.fetchone()
        assert after["recall_count"] == 0, "spontaneous must NOT increment recall_count"
        assert after["last_recalled_at"] is not None, "spontaneous must set last_recalled_at"
    finally:
        conn.close()


def test_recall_system_mode_does_not_reinforce(memory):
    """recall_mode='system' (default) leaves recall_count and last_recalled_at unchanged."""
    memory.save("システム想起テスト", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM observations WHERE content = %s AND person_id = %s",
                ("システム想起テスト", memory._person_id),
            )
            row = cur.fetchone()
        obs_id = row["id"]

        memory.recall("システム想起テスト", n=5, recall_mode="system")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT recall_count, last_recalled_at FROM observations WHERE id = %s",
                (obs_id,),
            )
            after = cur.fetchone()
        assert after["recall_count"] == 0
        assert after["last_recalled_at"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Time-decay scoring
# ---------------------------------------------------------------------------


def test_time_decay_prioritizes_recent_over_old(memory):
    """A recently-saved memory ranks higher than one backdated 60 days, all else equal."""
    memory.save("記憶古い", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE observations SET timestamp = now() - interval '60 days' "
                "WHERE content = %s AND person_id = %s",
                ("記憶古い", memory._person_id),
            )
        conn.commit()
    finally:
        conn.close()

    memory.save("記憶新しい", kind="observation", emotion="neutral")

    results = memory.recall("記憶", n=10, recall_mode="system")
    scores = {r["summary"]: r["score"] for r in results}

    assert "記憶新しい" in scores, "recent memory not found"
    assert "記憶古い" in scores, "old memory not found"
    assert scores["記憶新しい"] > scores["記憶古い"], (
        f"recent ({scores['記憶新しい']:.4f}) should exceed old ({scores['記憶古い']:.4f})"
    )


def test_recall_half_life_env_var(monkeypatch, memory):
    """RECALL_HALF_LIFE_DAYS env var is read via MemoryConfig."""
    from familiar_agent.config import MemoryConfig
    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "14.0")
    assert MemoryConfig().recall_half_life_days == pytest.approx(14.0)


def test_recall_time_floor_env_var(monkeypatch):
    """RECALL_TIME_FLOOR env var is read via MemoryConfig."""
    from familiar_agent.config import MemoryConfig
    monkeypatch.setenv("RECALL_TIME_FLOOR", "0.1")
    assert MemoryConfig().recall_time_floor == pytest.approx(0.1)
