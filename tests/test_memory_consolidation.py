"""Tests for Phase 2 memory consolidation features (PostgreSQL)."""

from __future__ import annotations

import os

import uuid
from datetime import datetime
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _encode_vector


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
    importance: float = 1.0,
) -> str:
    obs_id = str(uuid.uuid4())
    now = datetime.now()
    with mem._db_lock:
        conn = mem._ensure_connected()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id,content,timestamp,direction,kind,emotion,importance,person_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    obs_id, content, now,
                    "test", kind, emotion, importance, mem._person_id,
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
# Tests: importance column
# ---------------------------------------------------------------------------


def test_observations_has_importance_column() -> None:
    assert "importance" in _pg_columns("observations")


def test_importance_defaults_to_one() -> None:
    mem = _make_memory()
    obs_id = _insert_observation(mem, "default importance check")
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT importance FROM observations WHERE id=%s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    assert row is not None
    assert abs(row[0] - 1.0) < 1e-6


def test_decay_importance_reduces_old_records() -> None:
    mem = _make_memory()
    obs_id = _insert_observation(mem, "old memory", importance=1.0)
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("UPDATE observations SET timestamp='2025-01-01T00:00:00' WHERE id=%s", (obs_id,))
    conn.commit()
    conn.close()

    mem.decay_importance(before_date="2026-01-01", factor=0.95)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT importance FROM observations WHERE id=%s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    assert abs(row[0] - 0.95) < 1e-4


def test_decay_importance_skips_recent_records() -> None:
    from datetime import date
    mem = _make_memory()
    today = date.today().isoformat()
    obs_id = _insert_observation(mem, "fresh memory", importance=1.0)
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("UPDATE observations SET timestamp=%s WHERE id=%s", (today + "T12:00:00", obs_id))
    conn.commit()
    conn.close()

    mem.decay_importance(before_date=today, factor=0.95)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT importance FROM observations WHERE id=%s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    assert abs(row[0] - 1.0) < 1e-6


def test_decay_importance_is_cumulative() -> None:
    mem = _make_memory()
    obs_id = _insert_observation(mem, "old memory", importance=1.0)
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("UPDATE observations SET timestamp='2020-01-01T00:00:00' WHERE id=%s", (obs_id,))
    conn.commit()
    conn.close()

    mem.decay_importance(before_date="2026-01-01", factor=0.9)
    mem.decay_importance(before_date="2026-01-01", factor=0.9)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT importance FROM observations WHERE id=%s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    assert abs(row[0] - 0.81) < 1e-4


# ---------------------------------------------------------------------------
# Tests: superseded_by column
# ---------------------------------------------------------------------------


def test_observations_has_superseded_by_column() -> None:
    assert "superseded_by" in _pg_columns("observations")


def test_mark_superseded_sets_superseded_by() -> None:
    mem = _make_memory()
    old_id = _insert_observation(mem, "old version of memory")
    new_id = _insert_observation(mem, "updated version of memory")
    mem.mark_superseded(old_id=old_id, new_id=new_id)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT superseded_by FROM observations WHERE id=%s", (old_id,))
        row = cur.fetchone()
    conn.close()
    assert row[0] == new_id


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


def test_find_near_duplicates_returns_pairs() -> None:
    import numpy as np
    mem = _make_memory()

    vec = np.ones(1024, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    blob = _encode_vector(vec.tolist())

    id_a = _insert_observation(mem, "memory A about the living room")
    id_b = _insert_observation(mem, "memory B about the living room")

    conn = _pg_conn()
    with conn.cursor() as cur:
        for obs_id in (id_a, id_b):
            cur.execute(
                "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
                "ON CONFLICT (obs_id) DO UPDATE SET vector = EXCLUDED.vector",
                (obs_id, blob),
            )
    conn.commit()
    conn.close()

    pairs = mem.find_near_duplicates(threshold=0.95)
    pair_ids = {frozenset([p[0], p[1]]) for p in pairs}
    assert frozenset([id_a, id_b]) in pair_ids


def test_find_near_duplicates_failure_is_loud(caplog) -> None:
    """失敗は握り潰さず error＋トレースで残し、[] で degrade（棚卸し find_near_dup）。"""
    import logging
    from unittest.mock import MagicMock

    from familiar_agent.store.observations import ObservationStore

    store = ObservationStore.__new__(ObservationStore)
    ctx = MagicMock()
    ctx.conn.return_value.cursor.side_effect = RuntimeError("boom")
    store._ctx = ctx

    with caplog.at_level(logging.ERROR, logger="familiar_agent.store.observations"):
        result = store.find_near_duplicates()

    assert result == []
    assert any(
        r.levelno >= logging.ERROR
        and "find_near_duplicates failed" in r.getMessage()
        and r.exc_info
        for r in caplog.records
    )


def test_find_near_duplicates_skips_already_superseded() -> None:
    import numpy as np
    mem = _make_memory()

    vec = np.ones(1024, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    blob = _encode_vector(vec.tolist())

    id_a = _insert_observation(mem, "old memory X")
    id_b = _insert_observation(mem, "new memory X")
    mem.mark_superseded(old_id=id_a, new_id=id_b)

    conn = _pg_conn()
    with conn.cursor() as cur:
        for obs_id in (id_a, id_b):
            cur.execute(
                "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
                "ON CONFLICT (obs_id) DO UPDATE SET vector = EXCLUDED.vector",
                (obs_id, blob),
            )
    conn.commit()
    conn.close()

    pairs = mem.find_near_duplicates(threshold=0.95)
    pair_ids = {frozenset([p[0], p[1]]) for p in pairs}
    assert frozenset([id_a, id_b]) not in pair_ids
