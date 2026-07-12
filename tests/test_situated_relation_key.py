"""Tests for situated_embeddings.relation_key column (situated V2 schema器).

型つき関係エッジ（[D-在席相関/V2]）の器の第一歩。relation_key 列を追加し、
既存行は既定値 'presence' で埋まる。UNIQUE(obs_id, person_id) は据え置き、
生成・想起の挙動は変えない（列にデフォルトが入るだけ）。
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"

# situated_embeddings.vector は vector(1024)。非ゼロベクトルを入れる。
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _run_migration(conn) -> None:
    migration_path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-07-12-022_situated_relation_key.py"
    )
    spec = importlib.util.spec_from_file_location("situated_relation_key_migration", migration_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def test_migration_adds_relation_key_column() -> None:
    conn = _pg_conn()
    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'situated_embeddings'
              AND column_name = 'relation_key'
        """)
        cols = {r["column_name"] for r in cur.fetchall()}
    conn.close()

    assert cols == {"relation_key"}


def test_migration_defaults_relation_key_to_presence() -> None:
    """既存の書き込み経路（relation_key を指定しない INSERT）で 'presence' が入る。"""
    obs_id = str(uuid.uuid4())
    se_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
            "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
            (obs_id, "relation_key default test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
        )
        # relation_key を指定しない既存経路と同型の INSERT
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
            (se_id, obs_id, AGENT_SELF_ID, _VEC),
        )
        cur.execute(
            "SELECT relation_key FROM situated_embeddings WHERE id = %s", (se_id,)
        )
        row = cur.fetchone()
    conn.commit()
    conn.close()

    assert row["relation_key"] == "presence"


def test_unique_obs_person_still_holds() -> None:
    """UNIQUE(obs_id, person_id) は据え置き（V2 の複数行化はこのスライスでは入れない）。"""
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
            "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
            (obs_id, "unique test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
        )
    conn.commit()

    raised = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _VEC),
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        raised = True
        conn.rollback()
    conn.close()

    assert raised
