"""Tests for automatic PostgreSQL schema migrations on startup."""

from __future__ import annotations

import os

from pathlib import Path
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = os.environ["DATABASE_URL"]


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


def _pg_tables() -> set[str]:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
        tables = {r[0] for r in cur.fetchall()}
    conn.close()
    return tables


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


def test_auto_applies_migrations_on_first_connect() -> None:
    expected_ids = {p.stem for p in (Path.cwd() / "migration").glob("*.py")}

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
        mem.append_memory_event("memory.save", {"content": "x"}, queue_job=False)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM schema_migrations")
        applied = {r[0] for r in cur.fetchall()}
    conn.close()

    assert expected_ids.issubset(applied)
    tables = _pg_tables()
    assert {
        "observations",
        "obs_embeddings",
        "memory_events",
        "memory_jobs",
        "episodes",
        "episode_memories",
        "memory_activation",
        "unfinished_business",
        "relationship_state",
    }.issubset(tables)


def test_migrates_observations_has_all_columns() -> None:
    """All expected columns exist in the observations table after migration."""
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
        mem.append_memory_event("memory.save", {"content": "x"}, queue_job=False)

    cols = _pg_columns("observations")
    for name in ("kind", "emotion", "image_path", "image_data", "importance", "superseded_by"):
        assert name in cols, f"Missing column: {name}"


def test_migrations_are_idempotent_across_restarts() -> None:
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
        mem.append_memory_event("memory.save", {"content": "first"}, queue_job=False)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_migrations")
        count_first = cur.fetchone()[0]
    conn.close()

    # Reset singleton and reconnect — migrations must not re-run
    import familiar_agent.db as db_module
    with db_module._INSTANCE_LOCK:
        if db_module._INSTANCE is not None:
            try:
                db_module._INSTANCE.close()
            except Exception:
                pass
            db_module._INSTANCE = None

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem2 = ObservationMemory()
        mem2.append_memory_event("memory.save", {"content": "second"}, queue_job=False)

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_migrations")
        count_second = cur.fetchone()[0]
    conn.close()

    assert count_second == count_first
