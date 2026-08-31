"""Tests for the situated-correlation read layer (_read_observations_by_situated).

所有者絞り（observations.person_id）でなく situated 相関（situated_embeddings を
JOIN し s.person_id で person に紐づける）で観測を読む層。順序は timestamp DESC。
第一段では未接続で、既存の想起経路からは呼ばれない。
"""

from __future__ import annotations

import os

from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID


_DB_URL = os.environ["DATABASE_URL"]

_NOW = datetime(2026, 6, 1, 12, 0, 0)

# situated_embeddings.vector は vector(1024)。層はベクトルを使わない（timestamp 順）が
# 挿入には有効な非ゼロベクトルが要る（コサイン索引がゼロノルムを嫌う）。
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _insert_obs(
    cur, obs_id: str, content: str, kind: str, ts: datetime,
    emotion: str = "neutral", superseded_by: str | None = None,
) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, superseded_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, emotion, superseded_by),
    )


def _insert_situated(cur, se_id: str, obs_id: str, person_id: str) -> None:
    cur.execute(
        "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
        (se_id, obs_id, person_id, _VEC),
    )


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


# ── 1. situated 相関で person に紐づき、新しい順に返る ──────────────────────

def test_read_by_situated_returns_newest_first() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "old-1", "old", "conversation", _NOW - timedelta(hours=2))
        _insert_obs(cur, "mid-2", "mid", "conversation", _NOW - timedelta(hours=1))
        _insert_obs(cur, "new-3", "new", "conversation", _NOW)
        _insert_situated(cur, "se-old", "old-1", AGENT_SELF_ID)
        _insert_situated(cur, "se-mid", "mid-2", AGENT_SELF_ID)
        _insert_situated(cur, "se-new", "new-3", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 3, ("content", "timestamp"))

    assert [r["content"] for r in rows] == ["new", "mid", "old"]


def test_read_by_situated_respects_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        for i in range(5):
            _insert_obs(cur, f"c-{i}", f"row {i}", "conversation", _NOW + timedelta(minutes=i))
            _insert_situated(cur, f"se-{i}", f"c-{i}", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 3, ("content", "timestamp"))
    assert len(rows) == 3


# ── 2. 母集合の反証：所有者絞りでなく situated 相関である ──────────────────

def test_read_by_situated_includes_non_owner_when_correlated() -> None:
    """DEFAULT_PERSON_ID が所有する観測でも、AGENT_SELF_ID の situated 行があれば
    AGENT_SELF_ID の相関読み出しで返る（所有者絞りでないことの確認）。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "owned-by-other", "other owned", "conversation", _NOW)
        _insert_situated(cur, "se-corr", "owned-by-other", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",))

    assert [r["content"] for r in rows] == ["other owned"]


def test_read_by_situated_excludes_when_no_correlation_row() -> None:
    """situated 行が無い person_id では、所有していても返らない（相関が母集合を決める）。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "self-owned", "self owned", "conversation", _NOW)
        # AGENT_SELF_ID の situated 行は作らず、DEFAULT_PERSON_ID の行だけ作る
        _insert_situated(cur, "se-other", "self-owned", DEFAULT_PERSON_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",))
    assert rows == []


# ── 3. kind 絞り ────────────────────────────────────────────────────────────

def test_read_by_situated_filters_by_kind() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "ds-1", "a day summary", "day_summary", _NOW)
        _insert_obs(cur, "cv-1", "a conversation", "conversation", _NOW + timedelta(seconds=1))
        _insert_situated(cur, "se-ds", "ds-1", AGENT_SELF_ID)
        _insert_situated(cur, "se-cv", "cv-1", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",), kind="day_summary")
    assert [r["content"] for r in rows] == ["a day summary"]


# ── 4. keywords 絞り（content LIKE の OR）／空 keywords で全件 ───────────────

def test_read_by_situated_filters_by_keywords() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "kw-hit", "we talked about ramen today", "conversation", _NOW)
        _insert_obs(cur, "kw-miss", "nothing relevant here", "conversation", _NOW + timedelta(seconds=1))
        _insert_situated(cur, "se-hit", "kw-hit", AGENT_SELF_ID)
        _insert_situated(cur, "se-miss", "kw-miss", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",), keywords=("ramen",))
    assert [r["content"] for r in rows] == ["we talked about ramen today"]


def test_read_by_situated_empty_keywords_returns_all() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "a-1", "alpha", "conversation", _NOW)
        _insert_obs(cur, "b-1", "beta", "conversation", _NOW + timedelta(seconds=1))
        _insert_situated(cur, "se-a", "a-1", AGENT_SELF_ID)
        _insert_situated(cur, "se-b", "b-1", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",), keywords=())
    assert {r["content"] for r in rows} == {"alpha", "beta"}


# ── 5. columns 受け渡し（emotion 込み経路） ─────────────────────────────────

def test_read_by_situated_passes_through_emotion_column() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "emo-1", "happy row", "conversation", _NOW, emotion="happy")
        _insert_situated(cur, "se-emo", "emo-1", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content", "emotion"))
    assert rows[0]["emotion"] == "happy"


# ── 6. superseded_by が非 NULL の観測は除外 ────────────────────────────────

def test_read_by_situated_excludes_superseded() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "live-1", "live row", "conversation", _NOW)
        _insert_obs(cur, "dead-1", "superseded row", "conversation",
                    _NOW + timedelta(seconds=1), superseded_by="live-1")
        _insert_situated(cur, "se-live", "live-1", AGENT_SELF_ID)
        _insert_situated(cur, "se-dead", "dead-1", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",))
    assert [r["content"] for r in rows] == ["live row"]


# ── 7. 該当なしで空リスト ──────────────────────────────────────────────────

def test_read_by_situated_returns_empty_when_none() -> None:
    mem = _mem()
    rows = mem._observations._read_observations_by_situated(AGENT_SELF_ID, 10, ("content",))
    assert rows == []
