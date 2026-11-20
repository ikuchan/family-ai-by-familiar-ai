"""080（鍵の文の表 `obs_fact_keys` と、抜き済みの印 `obs_fact_runs` を足す・記-ad・2026-11-20・本人の承認）。

記録の本文から軽量LLM で抜いた事実（主語・関係・答え）を、問いに近い形の「鍵の文」に書き直して持つ。
事実 1 つから 2 行（答えを隠す向き、主語を隠す向き）。点は Ruri（768 次元）。記録が消えれば一緒に消える。
抜き済みの印は記録ごとに 1 行で、事実が 0 個だった記録にも付ける（もう一度呼ばないため）。
"""

from __future__ import annotations

import os
import uuid

import psycopg2
import psycopg2.extras
import pytest

_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["0.036"] * 768) + "]"


def _conn():
    return psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _cols(cur, table: str) -> dict:
    cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = %s ORDER BY ordinal_position",
        (table,),
    )
    return {r["column_name"]: r for r in cur.fetchall()}


def test_the_key_table_holds_a_768_vector_per_key_sentence():
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cols = _cols(cur, "obs_fact_keys")
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                "WHERE attrelid = 'obs_fact_keys'::regclass AND attname = 'vector'"
            )
            vtype = cur.fetchone()["t"]
            cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'obs_fact_keys'")
            defs = [r["indexdef"] for r in cur.fetchall()]
        assert set(cols) == {"id", "obs_id", "side", "sentence", "vector"}
        assert all(cols[c]["is_nullable"] == "NO" for c in ("obs_id", "side", "sentence", "vector"))
        assert vtype == "vector(768)"
        assert any("(obs_id)" in d for d in defs)  # 外部キーの列に索引
    finally:
        conn.rollback()
        conn.close()


def test_the_run_table_marks_each_record_once():
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cols = _cols(cur, "obs_fact_runs")
        assert set(cols) == {"obs_id", "model", "n_facts", "extracted_at"}
        assert cols["extracted_at"]["data_type"] == "timestamp with time zone"
    finally:
        conn.rollback()
        conn.close()


def test_side_is_one_of_the_two_directions():
    conn = _conn()
    obs_id = str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s, 'x', NOW(), '発話', 'neutral')",
                (obs_id,),
            )
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO obs_fact_keys (obs_id, side, sentence, vector) "
                    "VALUES (%s, 'どちらでもない', 's', %s::vector)",
                    (obs_id, _VEC),
                )
    finally:
        conn.rollback()
        conn.close()


def test_keys_and_marks_go_away_with_the_record():
    conn = _conn()
    obs_id = str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s, 'x', NOW(), '発話', 'neutral')",
                (obs_id,),
            )
            cur.execute(
                "INSERT INTO obs_fact_keys (obs_id, side, sentence, vector) "
                "VALUES (%s, '答え', '町内会の夏祭りの日', %s::vector)",
                (obs_id, _VEC),
            )
            cur.execute(
                "INSERT INTO obs_fact_runs (obs_id, model, n_facts) VALUES (%s, 'm', 1)", (obs_id,)
            )
            cur.execute("DELETE FROM observations WHERE id = %s", (obs_id,))
            cur.execute("SELECT count(*) AS c FROM obs_fact_keys WHERE obs_id = %s", (obs_id,))
            assert cur.fetchone()["c"] == 0
            cur.execute("SELECT count(*) AS c FROM obs_fact_runs WHERE obs_id = %s", (obs_id,))
            assert cur.fetchone()["c"] == 0
    finally:
        conn.rollback()
        conn.close()
