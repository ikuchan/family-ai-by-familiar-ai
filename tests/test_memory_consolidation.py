"""Tests for Phase 2 memory consolidation features (PostgreSQL)."""

from __future__ import annotations

import os

import uuid
from datetime import datetime
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory
from tests.hidden_helper import hidden_by


_DB_URL = os.environ["DATABASE_URL"]


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


def _make_memory() -> ObservationMemory:
    with patch.object(ObservationMemory._embedder.__class__, "pre_warm", lambda self: None):
        pass
    return ObservationMemory()


def _insert_observation(
    mem: ObservationMemory,
    content: str,
    kind: str = "observation",
    emotion: str = "neutral",
) -> str:
    obs_id = str(uuid.uuid4())
    now = datetime.now()
    with mem._db_lock:
        conn = mem._ensure_connected()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id,content,timestamp,direction,kind,emotion) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    obs_id,
                    content,
                    now,
                    "test",
                    kind,
                    emotion,
                ),
            )
        conn.commit()
    return obs_id


def _pg_columns(table: str) -> set[str]:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public'",
            (table,),
        )
        cols = {r[0] for r in cur.fetchall()}
    conn.close()
    return cols


# ---------------------------------------------------------------------------


def test_observations_has_no_superseded_by_column() -> None:
    """畳む印は関係へ移した（段 2）。列が残っていると古い道へ書けてしまう。"""
    assert "superseded_by" not in _pg_columns("observations")


def test_mark_superseded_records_the_new_side_in_a_relation() -> None:
    mem = _make_memory()
    old_id = _insert_observation(mem, "old version of memory")
    new_id = _insert_observation(mem, "updated version of memory")
    mem.mark_superseded(old_id=old_id, new_id=new_id)

    conn = _pg_conn()
    with conn.cursor() as cur:
        got = hidden_by(cur, old_id)
    conn.close()
    assert got == new_id


def test_recall_excludes_superseded_records() -> None:
    mem = _make_memory()
    old_id = _insert_observation(mem, "stale memory about cats")
    new_id = _insert_observation(mem, "updated memory about cats")
    mem.mark_superseded(old_id=old_id, new_id=new_id)
    results = mem.recall("cats", n=10, kind=None)
    returned_ids = {r["memory_id"] for r in results}
    assert old_id not in returned_ids


def test_recall_includes_non_superseded_records() -> None:
    mem = _make_memory()
    obs_id = _insert_observation(mem, "active memory about dogs")
    results = mem.recall("dogs", n=10, kind=None)
    returned_ids = {r["memory_id"] for r in results}
    assert obs_id in returned_ids


# ---------------------------------------------------------------------------
# Tests: near-duplicate detection
# ---------------------------------------------------------------------------
