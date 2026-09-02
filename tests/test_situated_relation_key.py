"""`situated_memories.relation_key`（型つき関係エッジの器）を確かめる。

022 が列を足し、023 が UNIQUE を `(obs_id, person_id, relation_key)` へ付け替えた。
**マイグレーションを流し直すのではなく、いまの姿を確かめる。** 044 で表を改名したので、
旧名を前提とした過去のマイグレーションはもう流せない（マイグレーションは一度しか
流れないので、後の版がスキーマを変えれば前の版は流し直せない）。
"""

from __future__ import annotations

import os

import uuid

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = os.environ["DATABASE_URL"]

# situated_memories.vector は vector(1024)。非ゼロベクトルを入れる。
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn



def test_the_relation_key_column_exists() -> None:
    conn = _pg_conn()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'situated_memories'
              AND column_name = 'relation_key'
        """)
        cols = {r["column_name"] for r in cur.fetchall()}
    conn.close()

    assert cols == {"relation_key"}


def test_relation_key_defaults_to_present() -> None:
    """relation_key を指定しない INSERT で 'present' が入る（047 で既定を改めた）。"""
    obs_id = str(uuid.uuid4())
    se_id = str(uuid.uuid4())
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
            "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
            (obs_id, "relation_key default test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
        )
        # relation_key を指定しない既存経路と同型の INSERT
        cur.execute(
            "INSERT INTO situated_memories (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
            (se_id, obs_id, AGENT_SELF_ID, _VEC),
        )
        cur.execute(
            "SELECT relation_key FROM situated_memories WHERE id = %s", (se_id,)
        )
        row = cur.fetchone()
    conn.commit()
    conn.close()

    assert row["relation_key"] == "present"


def test_the_same_relation_cannot_be_written_twice() -> None:
    """同じ面（obs_id, person_id, relation_key）は二度書けない。

    022 の段では `UNIQUE(obs_id, person_id)` が据え置きだったが、023 が
    `(obs_id, person_id, relation_key)` へ付け替えた。ここで確かめるのは**いまの姿**で、
    既定の `relation_key`（'presence'）どうしがぶつかることを見る。
    """
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
            "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
            (obs_id, "unique test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
        )
        cur.execute(
            "INSERT INTO situated_memories (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
        )
    conn.commit()

    raised = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        raised = True
        conn.rollback()
    conn.close()

    assert raised
