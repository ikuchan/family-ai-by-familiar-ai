"""Tests for append-only memory event log and pending job queue (PostgreSQL)."""

from __future__ import annotations

import json
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


def test_save_appends_event_and_pending_job() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save("hello world", kind="conversation", emotion="curious")

    conn = _pg_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT event_id, event_type, payload_json FROM memory_events")
        event = cur.fetchone()
        assert event is not None
        assert event["event_type"] == "memory.save"

        payload = json.loads(event["payload_json"])
        assert payload["content"] == "hello world"
        assert payload["kind"] == "conversation"
        assert payload["emotion"] == "curious"

        cur.execute(
            "SELECT job_type, status, attempts FROM memory_jobs WHERE event_id = %s",
            (event["event_id"],),
        )
        job = cur.fetchone()
        assert job is not None
        assert job["job_type"] == "materialize_observation"
        assert job["status"] == "pending"
        assert job["attempts"] == 0
    conn.close()


def test_save_with_dedupe_key_is_idempotent() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save("same payload", kind="conversation", dedupe_key="turn-1-conversation")
        assert mem.save("same payload", kind="conversation", dedupe_key="turn-1-conversation")

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM memory_events")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM memory_jobs")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM observations")
        assert cur.fetchone()[0] == 1
    conn.close()


def test_save_can_enqueue_without_immediate_materialization() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save("queued only", kind="conversation", materialize_now=False)

    conn = _pg_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM memory_events")
        assert cur.fetchone()["cnt"] == 1
        cur.execute("SELECT status FROM memory_jobs")
        job_row = cur.fetchone()
        assert job_row is not None
        assert job_row["status"] == "pending"
        cur.execute("SELECT COUNT(*) AS cnt FROM observations")
        assert cur.fetchone()["cnt"] == 0
    conn.close()


def test_save_continues_when_event_append_fails() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        with patch.object(mem, "append_memory_event", side_effect=RuntimeError("boom")):
            assert mem.save("still stored despite event failure")

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM memory_events")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM observations")
        assert cur.fetchone()[0] == 1
    conn.close()
