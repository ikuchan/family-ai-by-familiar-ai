"""Migration 035：時間軸の並べ替えの式に索引を張る。

`by_time` は `COALESCE(last_recalled_at, timestamp)` で絞って並べるが、この式に対する
索引が無く、全走査で答えていた。列ごとの索引（`idx_obs_timestamp`）は式には効かない。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-07-28-035_observations_recency_index.py"
_INDEX = "idx_observations_recency"


def _pg():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _run_migration(conn) -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)


def _indexdef(conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'observations' AND indexname = %s",
            (_INDEX,),
        )
        row = cur.fetchone()
    return None if row is None else str(row["indexdef"])


def test_migration_creates_the_index_on_the_ordering_expression():
    conn = _pg()
    try:
        _run_migration(conn)
        definition = _indexdef(conn)
        assert definition is not None, f"{_INDEX} が作られていない"
        # 並べ替えの鍵そのものに張られていること（列単体の索引では代わりにならない）。
        assert "COALESCE" in definition.upper()
        assert "last_recalled_at" in definition
        assert "timestamp" in definition
    finally:
        conn.close()


def test_migration_is_idempotent():
    """二度走らせても落ちない（適用済みの環境で再実行されうる）。"""
    conn = _pg()
    try:
        _run_migration(conn)
        _run_migration(conn)
        assert _indexdef(conn) is not None
    finally:
        conn.close()
