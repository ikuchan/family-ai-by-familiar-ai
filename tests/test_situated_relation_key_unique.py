"""Tests for situated V2 slice 2: UNIQUE(obs_id, person_id, relation_key) + upsert key化.

[D-在席相関/V2]：同一 (obs_id, person_id) に relation_key の違う複数行を許す。
生成はまだ 'presence' のみなので実挙動は不変だが、制約と upsert の同定キーを揃える。
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from unittest.mock import patch

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _run_migration(conn) -> None:
    migration_path = (
        Path(__file__).parent.parent / "migration" / "2026-07-12-023_situated_relation_key_unique.py"
    )
    spec = importlib.util.spec_from_file_location("situated_relation_key_unique_migration", migration_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def _insert_obs(cur, obs_id: str) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
        (obs_id, "slice2 test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
    )


def test_unique_includes_relation_key() -> None:
    """同一 (obs_id, person_id) でも relation_key が違えば2行入る。同一 triple は違反。"""
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        _insert_obs(cur, obs_id)
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector, relation_key) "
            "VALUES (%s, %s, %s, %s, 'presence')",
            (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector, relation_key) "
            "VALUES (%s, %s, %s, %s, 'speaker')",
            (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT relation_key FROM situated_embeddings WHERE obs_id=%s AND person_id=%s "
            "ORDER BY relation_key",
            (obs_id, AGENT_SELF_ID),
        )
        keys = [r["relation_key"] for r in cur.fetchall()]
    assert keys == ["presence", "speaker"]

    # 同一 triple は違反
    raised = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_embeddings (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s, %s, %s, %s, 'presence')",
                (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        raised = True
        conn.rollback()
    conn.close()
    assert raised


def test_upsert_separate_row_per_relation_key() -> None:
    """_upsert_situated_embedding が relation_key ごとに別行を作り、同 key 再upsertは更新。"""
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        _insert_obs(cur, obs_id)
    conn.commit()

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()

    vec = np.ones(1024, dtype=np.float32)
    mem._situated._upsert_situated_embedding(conn, obs_id, AGENT_SELF_ID, vec, relation_key="presence")
    mem._situated._upsert_situated_embedding(conn, obs_id, AGENT_SELF_ID, vec, relation_key="speaker")
    mem._situated._upsert_situated_embedding(conn, obs_id, AGENT_SELF_ID, vec, relation_key="presence")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT relation_key FROM situated_embeddings WHERE obs_id=%s AND person_id=%s "
            "ORDER BY relation_key",
            (obs_id, AGENT_SELF_ID),
        )
        keys = [r["relation_key"] for r in cur.fetchall()]
    conn.close()
    assert keys == ["presence", "speaker"]
