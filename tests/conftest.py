"""Global pytest configuration for familiar-ai."""

from __future__ import annotations

import os
import time

# Must be set before any familiar_agent imports so Database singleton picks up correct URL.
os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")

# 並列実行（pytest-xdist）ではワーカーごとに別 DB を使う。autouse の clean_db が
# 共有テーブルを truncate するため、1 DB を並列共有すると互いにデータを消し合う。
# `PYTEST_XDIST_WORKER`（gw0/gw1…）を DB 名に反映し、非並列（未設定/master）は従来の
# `familiar_test` を使う。DATABASE_URL は Database singleton が import 時に拾うので、
# familiar_agent の import より前にここで確定させる。
_PG_HOST = "postgresql://familiar:familiar@localhost:5433"
_BASE_DB = "familiar_test"
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
_DB_NAME = f"{_BASE_DB}_{_WORKER}" if (_WORKER and _WORKER != "master") else _BASE_DB
os.environ["DATABASE_URL"] = f"{_PG_HOST}/{_DB_NAME}"

import psycopg2  # noqa: E402
import pytest  # noqa: E402

_TEST_DB_URL = os.environ["DATABASE_URL"]


def _ensure_worker_db_and_schema() -> None:
    """ワーカー別 DB を用意し schema を張る（非並列の base DB は作成不要）。

    無ければ `familiar_test` へ管理接続して `CREATE DATABASE`（familiar は CREATEDB 権限）。
    並列起動時は複数ワーカーが同時に template1 を触って `being accessed` になり得るので
    retry する。作成後 apply_migrations で全 schema を張る（直接 psycopg2 で引くテストが
    最初に来ても表があるように、遅延適用でなくここで確定させる）。
    """
    if _DB_NAME != _BASE_DB:
        for attempt in range(10):
            try:
                admin = psycopg2.connect(f"{_PG_HOST}/{_BASE_DB}")
                admin.autocommit = True
                with admin.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (_DB_NAME,))
                    if not cur.fetchone():
                        cur.execute(f'CREATE DATABASE "{_DB_NAME}"')
                admin.close()
                break
            except psycopg2.Error:
                time.sleep(0.3 * (attempt + 1))
        else:
            return  # 作成できなければ諦める（そのワーカーの DB テストは失敗する）
    from familiar_agent.db_migrations import apply_migrations, default_migration_dir

    conn = psycopg2.connect(_TEST_DB_URL)
    apply_migrations(conn, default_migration_dir())
    conn.commit()
    conn.close()


try:
    _ensure_worker_db_and_schema()
except psycopg2.Error:
    # DB が起動していないときは収集ごと落とさない（DB を使わない純テストは走れる）。
    # DB を使うテストは autouse clean_db／各テストの接続で従来どおり失敗する。
    pass

# Reserved person IDs (mirrors migration 010)
_AGENT_SELF_ID = "00000000-0000-0000-0000-000000000000"
_DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"

_TRUNCATE_TABLES = [
    "situated_embeddings",
    "obs_embeddings",
    "memory_links",
    "episode_memories",
    "memory_salience",
    "unfinished_business",
    "memory_revisions",
    "behavior_policies",
    "semantic_facts",
    "memory_jobs",
    "memory_events",
    "episodes",
    "pending_speech",
    "observations",
    "relationship_state",
    "persons",
    "mental_state_log",
    "self_narrative_log",
    "agent_state",
]


def _reset_db_singleton() -> None:
    """Close and clear the Database singleton so the next test gets a fresh connection."""
    try:
        import familiar_agent.db as db_module
        with db_module._INSTANCE_LOCK:
            if db_module._INSTANCE is not None:
                try:
                    db_module._INSTANCE.close()
                except Exception:
                    pass
                db_module._INSTANCE = None
    except Exception:
        pass


def _truncate_all() -> None:
    """Truncate all test data tables. Silently skips tables that don't exist yet."""
    try:
        conn = psycopg2.connect(_TEST_DB_URL)
        conn.autocommit = True  # each statement is its own transaction
        with conn.cursor() as cur:
            for table in _TRUNCATE_TABLES:
                try:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                except Exception:
                    pass  # table may not exist yet
            # Re-insert reserved persons removed by TRUNCATE CASCADE
            now = "2026-01-01T00:00:00"
            for pid, name, display in [
                (_AGENT_SELF_ID, "__self__", "Agent self"),
                (_DEFAULT_PERSON_ID, "default", "Default Person"),
            ]:
                try:
                    cur.execute(
                        "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (pid, name, display, now, now),
                    )
                except Exception:
                    pass
        conn.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_db():
    """Isolate each test: reset singleton + truncate tables before and after."""
    _reset_db_singleton()
    _truncate_all()
    yield
    _reset_db_singleton()  # close open transactions before TRUNCATE
    _truncate_all()
