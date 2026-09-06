"""058（MI 間の関係の器）が表と索引を作っていることを確かめる。

記録どうしの関係を、二項の列から**多項の関係**へ移す（`設計方針_MI間の関係` v0.1）。
`superseded_by` は二つの記録しか結べないので、問いと版と答えと要約からなる一つの
やりとりを、一つの関係として指せなかった。ヘッダ（`relations`）と項（`relation_members`）
に分けると、項を何個でも持てて、`position` で順序も持てる。

段 1 は器だけを置く。既存の経路からは呼ばないので挙動は変わらない。`superseded_by` を
落とすのは段 2 である。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _columns(cur, table: str) -> dict[str, dict]:
    cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = %s",
        (table,),
    )
    return {r["column_name"]: r for r in cur.fetchall()}


def test_relations_table_has_kind_and_created_at():
    with _conn() as conn, conn.cursor() as cur:
        cols = _columns(cur, "relations")
        assert cols, "relations 表が無い"
        assert cols["id"]["data_type"] == "bigint"
        assert cols["kind"]["is_nullable"] == "NO"
        assert cols["created_at"]["is_nullable"] == "NO"


def test_relation_members_holds_obs_role_and_position():
    with _conn() as conn, conn.cursor() as cur:
        cols = _columns(cur, "relation_members")
        assert cols, "relation_members 表が無い"
        assert cols["relation_id"]["data_type"] == "bigint"
        assert cols["obs_id"]["is_nullable"] == "NO"
        assert cols["role"]["is_nullable"] == "NO"
        # 順序を持たない関係（共起）では位置が空く。
        assert cols["position"]["is_nullable"] == "YES"


def test_the_member_key_is_relation_and_obs_and_role():
    """同じ観測が同じ関係の同じ役割で二度入る形は無いので、主キーで落とす。"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = 'relation_members'::regclass AND i.indisprimary"
        )
        assert {r["attname"] for r in cur.fetchall()} == {"relation_id", "obs_id", "role"}


def test_the_lookups_have_indexes():
    """観測から関係を引く道と、種類で絞る道に索引がある。"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename IN ('relations', 'relation_members')"
        )
        names = {r["indexname"] for r in cur.fetchall()}
        assert "idx_relation_members_obs" in names
        assert "idx_relations_kind" in names


def test_dropping_a_relation_takes_its_members_with_it():
    """項はヘッダに従属する。ヘッダを消して項が残ると、どこにも属さない項が溜まる。"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO relations (kind) VALUES ('検査') RETURNING id")
        rid = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO relation_members (relation_id, obs_id, role, position) "
            "VALUES (%s, 'obs-1', '前', 0)",
            (rid,),
        )
        cur.execute("DELETE FROM relations WHERE id = %s", (rid,))
        cur.execute("SELECT count(*) AS n FROM relation_members WHERE relation_id = %s", (rid,))
        assert cur.fetchone()["n"] == 0
