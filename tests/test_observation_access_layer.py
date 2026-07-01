"""Tests for the thin observation access layer (_read_observations_by_kind)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"

_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _insert_obs(cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", person_id),
    )


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


# ── 1. _read_observations_by_kind: 順序と件数制限 ────────────────────────────

def test_read_observations_by_kind_returns_newest_first() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "old-1", "old curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=2))
        _insert_obs(cur, "mid-2", "mid curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "new-3", "new curiosity", "curiosity", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind("curiosity", AGENT_SELF_ID, 3, ("content", "timestamp"))

    assert len(rows) == 3
    assert rows[0]["content"] == "new curiosity"
    assert rows[1]["content"] == "mid curiosity"
    assert rows[2]["content"] == "old curiosity"


def test_read_observations_by_kind_respects_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        for i in range(5):
            _insert_obs(cur, f"c-{i}", f"curiosity {i}", "curiosity", AGENT_SELF_ID,
                        _NOW + timedelta(minutes=i))
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind("curiosity", AGENT_SELF_ID, 3, ("content", "timestamp"))

    assert len(rows) == 3


# ── 2. kind と person_id でフィルタされること ──────────────────────────────

def test_read_observations_by_kind_filters_by_kind() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "obs-1", "curiosity row", "curiosity", AGENT_SELF_ID, _NOW)
        _insert_obs(cur, "obs-2", "observation row", "observation", AGENT_SELF_ID, _NOW + timedelta(seconds=1))
        _insert_obs(cur, "obs-3", "feeling row", "feeling", AGENT_SELF_ID, _NOW + timedelta(seconds=2))
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert len(rows) == 1
    assert rows[0]["content"] == "curiosity row"


def test_read_observations_by_kind_filters_by_person_id() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "self-1", "agent curiosity", "curiosity", AGENT_SELF_ID, _NOW)
        _insert_obs(cur, "user-1", "user curiosity", "curiosity", DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1))
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert len(rows) == 1
    assert rows[0]["content"] == "agent curiosity"


def test_read_observations_by_kind_returns_empty_when_none_match() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "obs-1", "some feeling", "feeling", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert rows == []


# ── 3. recall_curiosities の付け替え後の戻り値の形 ──────────────────────────

def test_recall_curiosities_returns_expected_shape() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "cur-1", "first curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "cur-2", "second curiosity", "curiosity", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    result = mem.recall_curiosities(n=5)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time"}
    assert newest["summary"] == "second curiosity"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"


def test_recall_curiosities_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_curiosities(n=5)
    assert result == []
