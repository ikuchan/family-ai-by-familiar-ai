"""079（Ruri の数値の表 `obs_ruri_embeddings` を足す・記-y・2026-11-03・本人の決定）。

Ruri は記録の本文から数値を作るので、記録 1 件に 1 つ。記録の番号を鍵にし、記録が消えれば一緒に消える
（`obs_embeddings` と同じ形）。近い順に引く索引を付ける。
"""

from __future__ import annotations

import os
import uuid

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    return psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def test_the_table_has_a_768_vector_keyed_by_the_record():
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, udt_name, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'obs_ruri_embeddings' ORDER BY ordinal_position"
            )
            cols = {r["column_name"]: r for r in cur.fetchall()}
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = 'obs_ruri_embeddings'::regclass AND attname = 'vector'"
            )
            vtype = cur.fetchone()["t"]
            cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'obs_ruri_embeddings'")
            defs = [r["indexdef"] for r in cur.fetchall()]
        assert set(cols) == {"obs_id", "vector"}
        assert cols["vector"]["is_nullable"] == "NO"
        assert vtype == "vector(768)"
        assert any("hnsw" in d and "vector_cosine_ops" in d for d in defs)
        assert any("(obs_id)" in d and "UNIQUE" in d for d in defs)
    finally:
        conn.rollback()
        conn.close()


def test_rows_go_away_with_the_record():
    conn = _conn()
    obs_id = str(uuid.uuid4())
    vec = "[" + ",".join(["0.036"] * 768) + "]"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s, 'x', NOW(), '発話', 'neutral')",
                (obs_id,),
            )
            cur.execute(
                "INSERT INTO obs_ruri_embeddings (obs_id, vector) VALUES (%s, %s::vector)",
                (obs_id, vec),
            )
            cur.execute("DELETE FROM observations WHERE id = %s", (obs_id,))
            cur.execute(
                "SELECT count(*) AS c FROM obs_ruri_embeddings WHERE obs_id = %s", (obs_id,)
            )
            assert cur.fetchone()["c"] == 0
    finally:
        conn.rollback()
        conn.close()
