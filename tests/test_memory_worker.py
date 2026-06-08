"""Tests for background memory job worker (PostgreSQL)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.memory_worker import MemoryJobWorker, MemoryWorkerConfig
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


@pytest.mark.asyncio
async def test_worker_materializes_memory_save_jobs() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        payload = {
            "content": "queued memory",
            "direction": "unknown",
            "kind": "conversation",
            "emotion": "neutral",
            "image_path": None,
            "override_date": None,
        }
        event_id, created = mem.append_memory_event("memory.save", payload, queue_job=True)
        assert created is True
        assert event_id is not None

        worker = MemoryJobWorker(mem, MemoryWorkerConfig(batch_size=4, retry_delay_sec=0.0))
        processed = await worker.run_once()
        assert processed == 1

    conn = _pg_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT content FROM observations WHERE id = %s", (event_id,))
        obs = cur.fetchone()
        cur.execute("SELECT status, attempts FROM memory_jobs")
        job = cur.fetchone()
    conn.close()

    assert obs is not None
    assert obs["content"] == "queued memory"
    assert job["status"] == "done"
    assert job["attempts"] == 1


@pytest.mark.asyncio
async def test_worker_retries_then_dead_letters_failed_jobs() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        event_id, created = mem.append_memory_event(
            "memory.unknown",
            {"content": "bad event"},
            queue_job=True,
        )
        assert created is True
        assert event_id is not None

        worker = MemoryJobWorker(
            mem,
            MemoryWorkerConfig(batch_size=2, retry_delay_sec=0.0, max_attempts=2),
        )
        first = await worker.run_once()
        second = await worker.run_once()
        assert first == 1
        assert second == 1

    conn = _pg_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status, attempts, last_error FROM memory_jobs")
        row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row["status"] == "dead_letter"
    assert row["attempts"] == 2
    assert row["last_error"]


@pytest.mark.asyncio
async def test_run_loop_survives_claim_error() -> None:
    """_run_loop must not die when claim_pending_jobs raises an exception."""
    call_count = 0

    async def fake_run_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TypeError("tuple indices must be integers or slices, not str")
        # Stop the loop after a successful second call
        raise asyncio.CancelledError

    mem = MagicMock(spec=ObservationMemory)
    worker = MemoryJobWorker(mem, MemoryWorkerConfig(retry_delay_sec=0.0))
    worker.run_once = fake_run_once  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await worker._run_loop()

    assert call_count == 2, "worker should have retried after the first error"
