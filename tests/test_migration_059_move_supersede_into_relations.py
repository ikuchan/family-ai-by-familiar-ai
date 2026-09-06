"""059（`superseded_by` を関係へ移す）が効いていることを確かめる。

記録どうしのつながりを多項の関係へ移す（`設計方針_MI間の関係` v0.3・段 2）。列は
二つの記録しか結べず、改訂と畳み込みと解決と前進という別々の意味が一つの列に乗って
いた。関係へ移すと、種類が理由を言い、**役割 `旧` が「もう現行ではない」を表す**。

既存の辺は、どの書き手が作ったかを行から判別できない。推測で分類せず `未分類` で移す。

列を落とす操作は戻せないので、`observations_superseded_by_backup` へ写してから落とす。
"""

from __future__ import annotations

import os
import uuid
from importlib import util

import psycopg2
import psycopg2.extras
import pytest
from psycopg2.errors import UniqueViolation

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-06-059_move_supersede_into_relations.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m059", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_the_column_is_gone():
    """列が残っていると、書き手が古い道へ書き続けても誰も気づかない。"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'observations' AND column_name = 'superseded_by'"
        )
        assert cur.fetchall() == []


def test_the_backup_table_keeps_what_the_column_held():
    """落とす前の値を控える。控えが無ければ、移し損ねても元へ戻せない。"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'observations_superseded_by_backup'"
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        assert cols == {"id", "superseded_by"}


def test_every_backed_up_edge_became_a_relation():
    """控えた行が1件残らず関係になっている。

    件数を全体で突き合わせない。ほかの検査も `未分類` の関係を書くので、全体の数は
    この移行と無関係に増える。控えた id そのもので引く。
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM observations_superseded_by_backup")
        backed = [r["id"] for r in cur.fetchall()]
        if not backed:
            return
        cur.execute(
            "SELECT count(DISTINCT m.obs_id) AS n FROM relation_members m "
            "JOIN relations r ON r.id = m.relation_id AND r.kind = '未分類' "
            "WHERE m.role = '旧' AND m.obs_id = ANY(%s)",
            (backed,),
        )
        assert cur.fetchone()["n"] == len(backed)


def test_moving_an_edge_makes_an_unclassified_relation():
    """移し替えの本体。旧と新が位置 0 と 1 で入る。"""
    mod = _module()
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    conn = _conn()
    assert mod.move_edges(conn, [(old, new)]) == 1
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.kind, m.obs_id, m.role, m.position FROM relations r "
            "JOIN relation_members m ON m.relation_id = r.id "
            "WHERE m.relation_id IN (SELECT relation_id FROM relation_members WHERE obs_id = %s) "
            "ORDER BY m.position",
            (old,),
        )
        rows = cur.fetchall()
    assert [r["kind"] for r in rows] == ["未分類", "未分類"]
    assert [(r["obs_id"], r["role"], r["position"]) for r in rows] == [
        (old, "旧", 0),
        (new, "新", 1),
    ]
    conn.close()


def test_an_observation_can_be_old_only_once():
    """先着が勝つ。二度目の解決が張り替えると、どの記録が解決したかが失われる。

    いまは列の `WHERE superseded_by IS NULL` が原子性を持っていた。部分一意索引が
    それを引き継ぐ。索引が無ければ二本目が黙って通る。
    """
    mod = _module()
    old = str(uuid.uuid4())
    conn = _conn()
    assert mod.move_edges(conn, [(old, str(uuid.uuid4()))]) == 1
    with pytest.raises(UniqueViolation):
        mod.move_edges(conn, [(old, str(uuid.uuid4()))])
    conn.close()


def test_the_partial_index_only_covers_the_old_role():
    """`新` や `項` は何度でも入る。索引が役割を絞っていなければ、共起が書けなくなる。"""
    with _conn() as conn, conn.cursor() as cur:
        obs = str(uuid.uuid4())
        for _ in range(2):
            cur.execute("INSERT INTO relations (kind) VALUES ('検査') RETURNING id")
            rid = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO relation_members (relation_id, obs_id, role, position) "
                "VALUES (%s, %s, '項', NULL)",
                (rid, obs),
            )
        cur.execute(
            "SELECT count(*) AS n FROM relation_members WHERE obs_id = %s AND role = '項'",
            (obs,),
        )
        assert cur.fetchone()["n"] == 2
