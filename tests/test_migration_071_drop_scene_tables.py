"""071（`scene_entities`・`scene_events` 表を落とす・環-ab の R-5・2026-09-28）。

`SceneTracker` の表。作るだけで誰も読まず、書く入口の呼び手も無かった。コードを先に外し（R-5a）、表はそのあとに落とす。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-28-071_drop_scene_tables.py"
_TABLES = ("scene_entities", "scene_events")


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m071", _PATH)
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
