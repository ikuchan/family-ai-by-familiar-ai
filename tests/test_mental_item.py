"""Tests for the MentalItem vessel (Phase 1 A-1)."""

from __future__ import annotations

import os

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
from familiar_agent.mood_register import MoodPAD
from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = os.environ["DATABASE_URL"]

_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _insert_obs(cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime,
                 groundedness_g0: float = 1.0, superseded_by: str | None = None) -> None:
    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion, groundedness_g0, superseded_by) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", groundedness_g0, superseded_by),
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
            groundedness_g0=0.7, superseded_by="sm-0",
        )
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind(
        kind="self_model",
        n=1,
        columns=("id", "content", "timestamp", "emotion", "superseded_by", "groundedness_g0"),
    )
    row = rows[0]

    item = _row_to_mental_item(row)

    assert item.id == "sm-1"
    assert item.content == "self model content"
    assert item.supersedes == "sm-0"
    assert item.activation == 0.7
    # PAD 列を SELECT していない行は、感情も高ぶりも分からない。中立で埋めない。
    assert item.emotion is None
    assert item.arousal is None
    assert item.drive is None
    assert item.vector is None


# ── 1b. Y: PAD 列を持つ行は MoodPAD として emotion に載る（純関数） ──────────

def test_row_to_mental_item_loads_pad_emotion() -> None:
    row = {
        "id": "x", "content": "c", "superseded_by": None, "groundedness_g0": 1.0,
        "emotion_p": 0.8, "emotion_pn": 0.15, "emotion_a": 0.55, "emotion_dom": 0.6,
    }
    item = _row_to_mental_item(row)
    assert item.emotion == MoodPAD(0.8, 0.15, 0.55, 0.6)


def test_row_to_mental_item_leaves_the_feeling_unset_when_absent() -> None:
    """以前は既定 0.5 で中立に埋めていた。**その性質は意図して捨てた。**

    `用語_略語一覧` の PI 項が「評価結果としての中立と、未評価の未設定とを区別する
    ため」評価前は未設定で持つと定めている。埋めるとその区別が消える。
    """
    row = {"id": "x", "content": "c", "superseded_by": None, "groundedness_g0": 1.0}
    item = _row_to_mental_item(row)
    assert item.emotion is None
    assert item.arousal is None


# ── 0. 新居 core/mental_item から引ける（境界R B1） ─────────────────────────

def test_importable_from_core_mental_item() -> None:
    from familiar_agent.core.mental_item import (
        MentalItem as CoreMI,
        PrimitiveMentalItem as CorePI,
        _row_to_mental_item as core_row,
    )
    from familiar_agent.tools.memory import (
        MentalItem as MemMI,
        PrimitiveMentalItem as MemPI,
        _row_to_mental_item as mem_row,
    )
    # tools.memory は再輸出なので同一オブジェクト。
    assert CoreMI is MemMI
    assert CorePI is MemPI
    assert core_row is mem_row


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
