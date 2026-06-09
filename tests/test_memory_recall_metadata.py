"""Tests for evidence-backed memory recall metadata (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


def test_recall_semantic_includes_evidence_metadata() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory()
        assert mem.save("saw a cat by the window", kind="observation", emotion="curious")
        rows = mem.recall("cat", n=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["retrieval_method"] == "semantic"
    assert row["memory_id"]
    assert row["timestamp"]
    assert row["source_kind"] == "observation"
    assert "score" in row
    assert 0.0 <= float(row["confidence"]) <= 1.0


def test_recall_fallback_includes_metadata_and_low_confidence() -> None:
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
        mem.append_memory_event("memory.save", {"content": "seed"}, queue_job=False)

        # Insert a row directly (no embedding → forces recency fallback)
        now = datetime.now()
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id,content,timestamp,date,time,direction,kind,emotion,person_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    "legacy-row-1",
                    "older memory without embedding",
                    now.isoformat(),
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M"),
                    "unknown",
                    "conversation",
                    "neutral",
                    DEFAULT_PERSON_ID,
                ),
            )
        conn.commit()
        conn.close()

        rows = mem.recall("", n=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["retrieval_method"] == "recency"
    assert row["memory_id"] == "legacy-row-1"
    assert row["source_kind"] == "conversation"
    assert float(row["confidence"]) <= 0.55


def test_format_for_context_includes_confidence_guidance() -> None:
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    text = mem.format_for_context(
        [
            {
                "memory_id": "abcde12345",
                "date": "2026-03-03",
                "time": "10:00",
                "direction": "unknown",
                "source_kind": "conversation",
                "emotion": "neutral",
                "summary": "something uncertain happened",
                "confidence": 0.3,
            }
        ]
    )

    assert "conf<0.55" in text
    assert "id:abcde123" in text
    assert "low-confidence" in text
