"""060（共起を関係へ移す）が効いていることを確かめる。

記録どうしのつながりを表す独立した表を無くし、正本を一つにする
（`設計方針_MI間の関係` 段 5）。`wr_records` と `wr_record_items`（030）はヘッダと項に
分ける形を先に取っていたので、種類 `共起`・役割 `項` の関係へそのまま移せる。

`wr_record_items.mi_id` は名前に反して**観測の id** である（拡散想起が
`o.id::text = i.mi_id` で結合していた）。`relation_members.obs_id` と同じ粒度なので、
粒度を揃えたまま移せる。

元の2表は落とさず改名して残す。落とす操作は戻せないためである。
"""

from __future__ import annotations

import os
import uuid
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-07-060_move_wr_into_relations.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m060", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _tables() -> set:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        return {r["table_name"] for r in cur.fetchall()}


def test_the_original_tables_are_kept_under_a_new_name():
    """落とさず改名して残す。落とす操作は戻せない。"""
    names = _tables()
    assert "wr_records_moved" in names
    assert "wr_record_items_moved" in names
    assert "wr_records" not in names
    assert "wr_record_items" not in names


def test_moving_a_record_makes_a_cooccurrence_relation():
    """移し替えの本体。項は全部入り、位置は空く（共起に前後は無い）。"""
    mod = _module()
    ids = [str(uuid.uuid4()) for _ in range(3)]
    conn = _conn()
    assert mod.move_records(conn, [ids]) == 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.kind, m.obs_id, m.role, m.position FROM relations r "
            "JOIN relation_members m ON m.relation_id = r.id "
            "WHERE r.id = (SELECT relation_id FROM relation_members WHERE obs_id = %s)",
            (ids[0],),
        )
        rows = cur.fetchall()
    assert {r["kind"] for r in rows} == {"共起"}
    assert {r["obs_id"] for r in rows} == set(ids)
    assert all(r["role"] == "項" for r in rows)
    assert all(r["position"] is None for r in rows)
    conn.close()


def test_an_empty_record_is_not_moved():
    """項の無い WR は関係にならない（項の無い関係は関係ではない）。"""
    mod = _module()
    conn = _conn()
    assert mod.move_records(conn, [[]]) == 0
    conn.close()
