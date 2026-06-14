"""Tests for migration 016: observations timestamp normalization.

After migration 016:
- observations.timestamp is TIMESTAMPTZ (not TEXT)
- date and time columns are absent from the schema
- observations with non-ISO timestamps are rejected BEFORE any ALTER
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg2
import psycopg2.extras
import pytest


def _load_migration_016():
    path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-06-14-016_normalize_observation_timestamps.py"
    )
    spec = importlib.util.spec_from_file_location("migration_016", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fresh_conn():
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://familiar_ai:familiar_ai@localhost:5433/familiar_test",
    )
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# Schema verification against the already-migrated test DB
# ---------------------------------------------------------------------------


def test_timestamp_column_is_timestamptz():
    """observations.timestamp must be TIMESTAMPTZ after migration 016."""
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'observations'
                  AND column_name = 'timestamp'
            """)
            row = cur.fetchone()
        assert row is not None, "observations.timestamp column not found"
        assert row["data_type"] == "timestamp with time zone", (
            f"unexpected data_type: {row['data_type']}"
        )
    finally:
        conn.close()


def test_date_column_is_absent():
    """observations.date column must be dropped by migration 016."""
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) AS cnt
                FROM information_schema.columns
                WHERE table_name = 'observations'
                  AND column_name = 'date'
            """)
            row = cur.fetchone()
        assert row["cnt"] == 0, "observations.date should not exist after migration 016"
    finally:
        conn.close()


def test_time_column_is_absent():
    """observations.time column must be dropped by migration 016."""
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) AS cnt
                FROM information_schema.columns
                WHERE table_name = 'observations'
                  AND column_name = 'time'
            """)
            row = cur.fetchone()
        assert row["cnt"] == 0, "observations.time should not exist after migration 016"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pre-check guard: bad TEXT timestamps must raise before any DDL
# ---------------------------------------------------------------------------


def test_migration_rejects_bad_timestamp():
    """_check_timestamps() raises RuntimeError on non-ISO timestamp values.

    Uses a TEMP TABLE named 'observations' to shadow the permanent table in
    the same session — the pre-check SELECT hits our controlled TEXT data.
    Transaction is rolled back so the permanent table is never touched.
    """
    mod = _load_migration_016()

    conn = _fresh_conn()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE observations (
                    id        TEXT PRIMARY KEY,
                    timestamp TEXT,
                    person_id TEXT
                )
            """)
            cur.execute(
                "INSERT INTO observations VALUES (%s, %s, %s)",
                ("bad-ts", "not-a-timestamp", "test-person"),
            )

        with pytest.raises(RuntimeError, match="non-ISO"):
            mod._check_timestamps(conn)

    finally:
        conn.rollback()
        conn.close()
