"""BUG-1: Tests for utterance duplicate suppression (A) and purge migration (B)."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_DB_URL = os.environ["DATABASE_URL"]


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _make_memory() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _obs_count(conn, person_id: str, content: str, kind: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM observations "
            "WHERE person_id=%s AND content=%s AND kind=%s AND superseded_by IS NULL",
            (person_id, content, kind),
        )
        return cur.fetchone()["n"]


def _insert_obs_at(conn, person_id: str, content: str, kind: str, ts: datetime) -> str:
    """Direct INSERT bypassing dedup logic — used to plant fixtures."""
    obs_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations "
            "(id,content,timestamp,direction,kind,emotion,person_id,writer_id,subject_id,"
            " participants_json,scope) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, content, ts, "unknown", kind, "neutral",
             person_id, person_id, person_id, "[]", "speaker"),
        )
    conn.commit()
    return obs_id


# ── A: 書き込み側 dedup ──────────────────────────────────────


def test_dedup_skips_same_content_kind_within_window() -> None:
    """Same (person_id, content, kind) within window → only 1 row inserted."""

    mem = _make_memory()
    pid = mem._person_id
    content = f"重複テスト_{uuid.uuid4()}"

    # 窓の値は呼び出し時に MemoryConfig から読む（層は設定を持たない）。
    original_window = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "30"
    try:
        ok1, _ = mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
        ok2, _ = mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
        ok3, _ = mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
    finally:
        if original_window is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original_window

    conn = _pg_conn()
    n = _obs_count(conn, pid, content, "utterance")
    conn.close()
    assert n == 1, f"Expected 1 row after dedup, got {n}"


def test_dedup_allows_different_content() -> None:
    """Different content with same kind must both be inserted."""

    mem = _make_memory()
    pid = mem._person_id
    suffix = uuid.uuid4()
    c1 = f"内容A_{suffix}"
    c2 = f"内容B_{suffix}"

    # 窓の値は呼び出し時に MemoryConfig から読む（層は設定を持たない）。
    original_window = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "30"
    try:
        mem.save_with_id(c1, kind="utterance", writer_id=pid, subject_id=pid)
        mem.save_with_id(c2, kind="utterance", writer_id=pid, subject_id=pid)
    finally:
        if original_window is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original_window

    conn = _pg_conn()
    n1 = _obs_count(conn, pid, c1, "utterance")
    n2 = _obs_count(conn, pid, c2, "utterance")
    conn.close()
    assert n1 == 1 and n2 == 1, f"Expected 1+1, got {n1}+{n2}"


def test_dedup_allows_different_kind() -> None:
    """Same content but different kind must both be inserted."""

    mem = _make_memory()
    pid = mem._person_id
    content = f"kindテスト_{uuid.uuid4()}"

    # 窓の値は呼び出し時に MemoryConfig から読む（層は設定を持たない）。
    original_window = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "30"
    try:
        mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
        mem.save_with_id(content, kind="conversation", writer_id=pid, subject_id=pid)
    finally:
        if original_window is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original_window

    conn = _pg_conn()
    n_u = _obs_count(conn, pid, content, "utterance")
    n_c = _obs_count(conn, pid, content, "conversation")
    conn.close()
    assert n_u == 1 and n_c == 1, f"utterance={n_u} conversation={n_c}"


def test_dedup_disabled_when_window_zero() -> None:
    """MEMORY_DEDUP_WINDOW_SECS=0 must allow duplicate inserts."""

    mem = _make_memory()
    pid = mem._person_id
    content = f"dedup無効テスト_{uuid.uuid4()}"

    # 窓の値は呼び出し時に MemoryConfig から読む（層は設定を持たない）。
    original_window = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "0"
    try:
        mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
        mem.save_with_id(content, kind="utterance", writer_id=pid, subject_id=pid)
    finally:
        if original_window is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original_window

    conn = _pg_conn()
    n = _obs_count(conn, pid, content, "utterance")
    conn.close()
    assert n == 2, f"Expected 2 rows with dedup disabled, got {n}"


# ── B: purge マイグレーション ─────────────────────────────────


def _run_purge_migration(conn) -> None:
    import importlib.util
    migration_path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-06-29-019_purge_utterance_duplicates.py"
    )
    spec = importlib.util.spec_from_file_location("purge_migration", migration_path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def test_purge_migration_removes_duplicate_embeddings() -> None:
    """Purge migration supersedes duplicates and removes their embeddings."""
    mem = _make_memory()
    pid = mem._person_id
    content = f"purgeテスト_{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc)

    conn = _pg_conn()
    # Plant 5 duplicates within 5 seconds
    ids = []
    for i in range(5):
        oid = _insert_obs_at(conn, pid, content, "utterance", now + timedelta(seconds=i))
        ids.append(oid)

    # Insert a fake situated_embedding for each (vector not important for this test)
    from familiar_agent.db import vec_to_sql
    import numpy as np
    fake_vec = vec_to_sql(np.zeros(1024).tolist())
    with conn.cursor() as cur:
        for oid in ids:
            cur.execute(
                "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) "
                "VALUES (%s, %s, %s, %s::vector) ON CONFLICT DO NOTHING",
                (str(uuid.uuid4()), oid, pid, fake_vec),
            )
    conn.commit()

    _run_purge_migration(conn)

    with conn.cursor() as cur:
        # Only one unsuperseded row should remain
        cur.execute(
            "SELECT COUNT(*) AS n FROM observations "
            "WHERE person_id=%s AND content=%s AND kind=%s AND superseded_by IS NULL",
            (pid, content, "utterance"),
        )
        remaining = cur.fetchone()["n"]

        # Superseded rows must have superseded_by set to the first id
        cur.execute(
            "SELECT COUNT(*) AS n FROM observations "
            "WHERE person_id=%s AND content=%s AND kind=%s AND superseded_by IS NOT NULL",
            (pid, content, "utterance"),
        )
        superseded = cur.fetchone()["n"]

        # Embeddings for superseded rows must be gone
        cur.execute(
            "SELECT COUNT(*) AS n FROM situated_embeddings WHERE obs_id = ANY(%s)",
            (ids[1:],),  # all but the first
        )
        leftover_emb = cur.fetchone()["n"]
    conn.close()

    assert remaining == 1, f"Expected 1 unsuperseded, got {remaining}"
    assert superseded == 4, f"Expected 4 superseded, got {superseded}"
    assert leftover_emb == 0, f"Expected 0 leftover embeddings, got {leftover_emb}"


def test_purge_migration_keeps_non_duplicates() -> None:
    """Purge migration must not touch observations that are not duplicates."""
    mem = _make_memory()
    pid = mem._person_id
    c1 = f"一意A_{uuid.uuid4()}"
    c2 = f"一意B_{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc)

    conn = _pg_conn()
    oid1 = _insert_obs_at(conn, pid, c1, "utterance", now)
    oid2 = _insert_obs_at(conn, pid, c2, "utterance", now + timedelta(seconds=1))
    conn.commit()

    _run_purge_migration(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT superseded_by FROM observations WHERE id = ANY(%s)",
            ([oid1, oid2],),
        )
        rows = cur.fetchall()
    conn.close()

    for row in rows:
        assert row["superseded_by"] is None, f"Non-duplicate was superseded: {row}"


def test_purge_migration_respects_60s_boundary() -> None:
    """Observations > 60 s apart with same content must not be merged."""
    mem = _make_memory()
    pid = mem._person_id
    content = f"時間境界テスト_{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc)

    conn = _pg_conn()
    # Two observations 90 seconds apart — should NOT be merged
    oid_early = _insert_obs_at(conn, pid, content, "utterance", now)
    oid_late  = _insert_obs_at(conn, pid, content, "utterance", now + timedelta(seconds=90))
    conn.commit()

    _run_purge_migration(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, superseded_by FROM observations WHERE id = ANY(%s)",
            ([oid_early, oid_late],),
        )
        rows = {r["id"]: r["superseded_by"] for r in cur.fetchall()}
    conn.close()

    assert rows[oid_early] is None, "Early observation was wrongly superseded"
    assert rows[oid_late]  is None, "Late observation (90s apart) was wrongly superseded"
