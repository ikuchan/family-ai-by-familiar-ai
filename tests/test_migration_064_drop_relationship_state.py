"""064（`relationship_state` 表を落とす・環-d・2026-09-15）。

書き手 `RelationshipTracker`（`relationship.py`）は毎ターン `record_conversation()` で書いていたが、
`trust`／`intimacy` を読むのは呼び手の無い `_select_addressee` だけだった。設計の「移管」は
やめて撤去（2026-09-15 決定）。関係は O の MI（関係の面・`人物` のまとめ）が担う。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-15-064_drop_relationship_state.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m064", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _has_table() -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.relationship_state') IS NOT NULL AS present")
        return bool(cur.fetchone()["present"])


def test_the_table_is_gone_from_the_schema():
    assert not _has_table()


def test_upgrade_drops_the_table_and_is_idempotent():
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS relationship_state (state_key text)")
    assert _has_table()
    _module().upgrade(conn)
    assert not _has_table()
    _module().upgrade(conn)
    assert not _has_table()
    conn.close()
