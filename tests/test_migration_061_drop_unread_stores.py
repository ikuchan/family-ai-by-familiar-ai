"""061（読み手の無いストアを落とす・環-d・2026-09-14）。

`self_narrative_log`（層 1 の畳み込みが代替）・`semantic_facts`／`behavior_policies`／
`memory_links`／`memory_revisions`（`legacy/semantic_layer.py` だけが書き、読む呼び手が無い）を
落とす。改名して残す作法（060）は取らない——バックアップを確かめたうえでの決定。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-14-061_drop_unread_stores.py"
DROPPED = (
    "self_narrative_log",
    "semantic_facts",
    "behavior_policies",
    "memory_links",
    "memory_revisions",
)


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m061", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _tables() -> set:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        return {r["table_name"] for r in cur.fetchall()}


def test_the_unread_stores_are_gone_from_the_schema():
    assert not (set(DROPPED) & _tables())


def test_upgrade_drops_the_tables_and_is_idempotent():
    mod = _module()
    conn = _conn()
    with conn.cursor() as cur:
        for t in DROPPED:  # 落とす前の形を最小限に再現する（列の中身は問わない）
            cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (id text)")
    assert set(DROPPED) <= _tables()
    mod.upgrade(conn)
    assert not (set(DROPPED) & _tables())
    mod.upgrade(conn)  # 二度目は何もしない
    assert not (set(DROPPED) & _tables())
    conn.close()


def test_the_parents_survive():
    # 落とす表は子（FK の参照元）だけ。親の `persons`・`observations` は残る。
    assert {"persons", "observations"} <= _tables()
