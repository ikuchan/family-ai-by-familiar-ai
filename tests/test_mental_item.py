"""Tests for the MentalItem vessel (Phase 1 A-1)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import (
    MentalItem,
    ObservationMemory,
    PrimitiveMentalItem,
    _EmbeddingModel,
    _row_to_mental_item,
)
from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"

_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _insert_obs(cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime,
                 importance: float = 1.0, superseded_by: str | None = None) -> None:
    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion, person_id, importance, superseded_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", person_id, importance, superseded_by),
    )


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


# ── 1. _row_to_mental_item builds a MentalItem from an observation row ──────

def test_row_to_mental_item_builds_from_row() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(
            cur, "sm-1", "self model content", "self_model", AGENT_SELF_ID, _NOW,
            importance=0.7, superseded_by="sm-0",
        )
    conn.close()

    mem = _mem()
    rows = mem._read_observations_by_kind(
        kind="self_model",
        person_id=AGENT_SELF_ID,
        n=1,
        columns=("id", "content", "timestamp", "emotion", "superseded_by", "importance"),
    )
    row = rows[0]

    item = _row_to_mental_item(row)

    assert item.id == "sm-1"
    assert item.content == "self model content"
    assert item.supersedes == "sm-0"
    assert item.activation == 0.7
    assert item.emotion is None
    assert item.drive is None
    assert item.vector is None


# ── 2. MentalItem inherits PrimitiveMentalItem ──────────────────────────────

def test_mental_item_inherits_primitive_mental_item() -> None:
    item = MentalItem()

    assert isinstance(item, PrimitiveMentalItem)
    assert item.emotion is None
    assert item.drive is None


# ── 3. recall_self_model preserves external behavior ────────────────────────

def test_recall_self_model_returns_same_shape_newest_first() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "sm-old", "old self model", "self_model", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "sm-new", "new self model", "self_model", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=2)

    assert len(result) == 2
    assert set(result[0].keys()) == {"summary", "date", "time", "emotion"}
    assert result[0]["summary"] == "new self model"
    assert result[1]["summary"] == "old self model"
