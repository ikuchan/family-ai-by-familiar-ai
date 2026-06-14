"""Global pytest configuration for familiar-ai."""

from __future__ import annotations

import os

# Must be set before any familiar_agent imports so Database singleton picks up correct URL.
os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")
os.environ["DATABASE_URL"] = "postgresql://familiar:familiar@localhost:5433/familiar_test"

import psycopg2  # noqa: E402
import pytest  # noqa: E402

_TEST_DB_URL = os.environ["DATABASE_URL"]

# Reserved person IDs (mirrors migration 010)
_AGENT_SELF_ID = "00000000-0000-0000-0000-000000000000"
_DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"

_TRUNCATE_TABLES = [
    "situated_embeddings",
    "obs_embeddings",
    "memory_links",
    "episode_memories",
    "memory_activation",
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
