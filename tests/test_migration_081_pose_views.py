"""081（定点ごとの前の見え `pose_views` を足す・知-i 段 2・2026-11-24・本人の承認）。

見えの普通（`pose_norms`）は 3 秒ごとに育つ移動平均なので、置き物が変わっても十数回で吸収される。前の見えは
それとは別に、見比べて変わったと判定したときだけ置き換える 1 枚を定点ごとに持つ。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def test_each_pose_keeps_one_earlier_view_with_its_picture():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'pose_views'"
            )
            cols = {r["column_name"]: r["is_nullable"] for r in cur.fetchall()}
            cur.execute(
                "SELECT attname, format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = 'pose_views'::regclass AND attnum > 0"
            )
            types = {r["attname"]: r["t"] for r in cur.fetchall()}
            cur.execute(
                "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid "
                "AND a.attnum = ANY(i.indkey) WHERE i.indrelid = 'pose_views'::regclass AND i.indisprimary"
            )
            pk = [r["attname"] for r in cur.fetchall()]
        assert set(cols) == {"pose_name", "embedding", "image", "seen_at", "checked_at"}
        assert pk == ["pose_name"]
        assert types["embedding"] == "vector(384)" and types["image"] == "bytea"
        assert all(cols[c] == "NO" for c in ("embedding", "image", "seen_at"))
        assert cols["checked_at"] == "YES"  # まだ見比べていない定点は空
    finally:
        conn.rollback()
        conn.close()
