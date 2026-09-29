"""072（`memory_salience`・`episode_memories`・`episodes` 表を落とす・環-ab D・2026-09-29）。

W を溜めていた旧方式の表（`memory_salience`）とエピソードの表。読み書きしていた口（`get_working_memory`・`refresh_working_memory`・`create_episode`・`append_to_episode`）は呼び手が無く、環-ab D で先に外した。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-29-072_drop_salience_and_episodes.py"
_TABLES = ("memory_salience", "episode_memories", "episodes")


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m072", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _present() -> list[str]:
    with _conn() as conn, conn.cursor() as cur:
        out = []
        for t in _TABLES:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS present", (f"public.{t}",))
            if cur.fetchone()["present"]:
                out.append(t)
        return out


def test_the_tables_are_gone_from_the_schema():
    assert _present() == []


def test_upgrade_drops_the_tables_and_is_idempotent():
    conn = _conn()
    with conn.cursor() as cur:
        for t in _TABLES:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (id text)")
    assert _present() == list(_TABLES)
    _module().upgrade(conn)
    assert _present() == []
    _module().upgrade(conn)
    assert _present() == []
    conn.close()
