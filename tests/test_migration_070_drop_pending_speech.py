"""070（`pending_speech` 表を落とす・環-ab の A・2026-09-28）。

書く口は主LLM に出していない記憶の道具 `note_to_share` だけで、読む口（保留を配る口）は出-as 段 9b で外した。
コード（`note_to_share`・`PendingSpeechStore`・`PendingSpeechConfig`）を先に外し（A-1・A-2）、表はそのあとに落とす。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-28-070_drop_pending_speech.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m070", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _has_table() -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.pending_speech') IS NOT NULL AS present")
        return bool(cur.fetchone()["present"])


def test_the_table_is_gone_from_the_schema():
    assert not _has_table()


def test_upgrade_drops_the_table_and_is_idempotent():
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS pending_speech (id text)")
    assert _has_table()
    _module().upgrade(conn)
    assert not _has_table()
    _module().upgrade(conn)
    assert not _has_table()
    conn.close()
