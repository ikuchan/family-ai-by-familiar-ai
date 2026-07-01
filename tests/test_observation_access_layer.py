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


# ── 4. recall_self_model の付け替え後の戻り値の形（emotion 込み経路） ────────

def _insert_obs_with_emotion(
    cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime, emotion: str
) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, emotion, person_id),
    )


def test_recall_self_model_returns_expected_shape_newest_first_with_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-1", "first self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=2), "neutral")
        _insert_obs_with_emotion(cur, "sm-2", "second self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "sm-3", "third self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "neutral")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=2)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time", "emotion"}
    assert newest["summary"] == "third self model"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"
    assert result[1]["summary"] == "second self model"


def test_recall_self_model_returns_distinct_emotion_values_unchanged() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-emo-1", "neutral self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "sm-emo-2", "happy self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "happy")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=5)

    by_summary = {r["summary"]: r["emotion"] for r in result}
    assert by_summary["happy self model"] == "happy"
    assert by_summary["neutral self model"] == "neutral"


def test_recall_self_model_filters_by_agent_self_id() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-self", "agent self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "neutral")
        _insert_obs_with_emotion(cur, "sm-other", "other person self model", "self_model",
                                  DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1), "neutral")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=10)

    assert len(result) == 1
    assert result[0]["summary"] == "agent self model"


def test_recall_self_model_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_self_model(n=5)
    assert result == []


# ── 5. recall_day_summaries の付け替え後の戻り値の形（在席者スコープ経路） ──

def test_recall_day_summaries_returns_expected_shape_newest_first_with_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "ds-1", "first day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=2), "neutral")
        _insert_obs_with_emotion(cur, "ds-2", "second day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "ds-3", "third day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW, "neutral")
    conn.close()

    mem = _mem()
    assert mem._person_id == DEFAULT_PERSON_ID
    result = mem.recall_day_summaries(n=2)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time", "emotion"}
    assert newest["summary"] == "third day summary"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"
    assert result[1]["summary"] == "second day summary"


def test_recall_day_summaries_scopes_to_this_memorys_person_id() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "ds-present", "present person day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW, "neutral")
        _insert_obs_with_emotion(cur, "ds-other", "other person day summary", "day_summary",
                                  AGENT_SELF_ID, _NOW + timedelta(seconds=1), "neutral")
    conn.close()

    mem = _mem()
    assert mem._person_id == DEFAULT_PERSON_ID
    result = mem.recall_day_summaries(n=10)

    assert len(result) == 1
    assert result[0]["summary"] == "present person day summary"


def test_recall_day_summaries_returns_distinct_emotion_values_unchanged() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "ds-emo-1", "neutral day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "ds-emo-2", "happy day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW, "happy")
    conn.close()

    mem = _mem()
    result = mem.recall_day_summaries(n=5)

    by_summary = {r["summary"]: r["emotion"] for r in result}
    assert by_summary["happy day summary"] == "happy"
    assert by_summary["neutral day summary"] == "neutral"


def test_recall_day_summaries_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_day_summaries(n=5)
    assert result == []
