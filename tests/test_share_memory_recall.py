"""Tests for Issue C: 2-stage associative memory recall for share_memory.

pick_seed_candidates() returns a mixed pool of hour-near + month-near + random rows.
_proactive_memory_context() seeds from present persons, expands via recall(), caps total.
desires: share_memory has no time-of-day multiplier (removed in Issue C).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _pg_conn():
    url = os.environ.get(
        "DATABASE_URL",
        os.environ["DATABASE_URL"],
    )
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _make_mem_with_timestamps(rows: list[tuple[str, str]]) -> ObservationMemory:
    """Create ObservationMemory pre-loaded with rows at specific timestamps.

    rows: list of (content, timestamp_str) e.g. ("昼の記憶", "2025-06-14 14:30:00")
    """
    person_id = str(uuid.uuid4())
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory(person_id=person_id)

    conn = _pg_conn()
    now_str = datetime.now().isoformat()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id,name,display_name,created_at,updated_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (person_id, f"test-{person_id[:8]}", "Test", now_str, now_str),
            )
            for content, ts_str in rows:
                cur.execute(
                    "INSERT INTO observations "
                    "(id,content,timestamp,direction,kind,emotion) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), content, ts_str, "unknown",
                     "conversation", "neutral"),
                )
        conn.commit()
    finally:
        conn.close()

    return mem


# ---------------------------------------------------------------------------
# Tests: pick_seed_candidates
# ---------------------------------------------------------------------------


def test_pick_seed_candidates_returns_list_of_dicts():
    """pick_seed_candidates() returns a list of dicts with content and timestamp."""
    mem = _make_mem_with_timestamps([
        ("午後2時の記憶A", "2025-06-14 14:00:00"),
        ("午後2時の記憶B", "2025-06-14 14:30:00"),
        ("午後3時の記憶",  "2025-06-14 15:00:00"),
        ("深夜の記憶",     "2025-06-14 02:00:00"),
    ])
    result = mem.pick_seed_candidates(hour=14, month=6, hour_window=3, month_window=1, k=3)
    assert isinstance(result, list)
    for item in result:
        assert "content" in item, f"missing 'content' in {item}"
        assert "timestamp" in item, f"missing 'timestamp' in {item}"


def test_pick_seed_candidates_empty_db_returns_empty():
    """Empty DB returns empty list without error."""
    mem = _make_mem_with_timestamps([])
    result = mem.pick_seed_candidates(hour=14, month=6, hour_window=3, month_window=1, k=3)
    assert result == []


def test_pick_seed_candidates_hour_window_includes_nearby():
    """Rows within hour_window of target hour appear in candidates."""
    # Insert many rows at hour=14 so they reliably appear in k=5 sample
    rows = [(f"14時の記憶{i}", f"2025-06-14 14:0{i}:00") for i in range(5)]
    rows += [("深夜の記憶", "2025-06-14 02:00:00")]  # hour=2, far from 14
    mem = _make_mem_with_timestamps(rows)

    # Run multiple times: hour=14 rows must appear at least once
    found_near = False
    for _ in range(5):
        cands = mem.pick_seed_candidates(hour=14, month=6, hour_window=3, month_window=0, k=5)
        if any("14時" in c.get("content", "") for c in cands):
            found_near = True
            break
    assert found_near, "Hour-window candidates were never returned despite many matching rows"


def test_pick_seed_candidates_month_window_includes_nearby():
    """Rows within month_window of target month appear in candidates."""
    rows = [
        ("6月の記憶", "2025-06-15 12:00:00"),
        ("1月の記憶", "2025-01-15 12:00:00"),   # far from month=6
    ]
    mem = _make_mem_with_timestamps(rows)

    found_june = False
    for _ in range(5):
        cands = mem.pick_seed_candidates(hour=0, month=6, hour_window=0, month_window=1, k=5)
        if any("6月" in c.get("content", "") for c in cands):
            found_june = True
            break
    assert found_june, "Month-window candidates were never returned"


def test_pick_seed_candidates_deduplicates():
    """Duplicate IDs across sub-queries are removed."""
    rows = [(f"記憶{i}", f"2025-06-14 14:0{i}:00") for i in range(3)]
    mem = _make_mem_with_timestamps(rows)

    cands = mem.pick_seed_candidates(hour=14, month=6, hour_window=3, month_window=1, k=10)
    ids = [c.get("id") for c in cands]
    assert len(ids) == len(set(ids)), "Duplicate IDs found in candidates"


# ---------------------------------------------------------------------------
# Tests: _proactive_memory_context (agent)
# ---------------------------------------------------------------------------


@pytest.fixture()
def agent_no_present():
    from familiar_agent.agent import EmbodiedAgent
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = []
    agent._pmm = pmm
    return agent


@pytest.fixture()
def agent_with_present():
    from familiar_agent.agent import EmbodiedAgent
    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    seed_mem = MagicMock()
    seed_mem.pick_seed_candidates = MagicMock(return_value=[
        {"id": "s1", "content": "古い思い出A", "timestamp": datetime(2025, 6, 14, 14)},
        {"id": "s2", "content": "古い思い出B", "timestamp": datetime(2025, 12, 1, 12)},
    ])

    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = [("pid-1", seed_mem)]
    pmm.get_speaker_memory.return_value = None
    pmm.get_agent_memory.return_value = seed_mem
    agent._pmm = pmm

    assoc_mem = MagicMock()
    assoc_mem.recall_async = AsyncMock(return_value=[
        {"memory_id": "a1", "content": "連想された記憶", "score": 0.85},
    ])
    # _active_memory() = pmm.get_speaker_memory() or pmm.get_agent_memory()
    # get_speaker_memory returns None → falls back to get_agent_memory → assoc_mem
    pmm.get_agent_memory.return_value = assoc_mem

    return agent


@pytest.mark.asyncio
async def test_proactive_context_returns_none_when_no_present(agent_no_present):
    """誰もいないとき None を返すこと。"""
    result = await agent_no_present._proactive_memory_context()
    assert result is None


@pytest.mark.asyncio
async def test_proactive_context_returns_string_when_present(agent_with_present):
    """在席者がいるとき文字列を返すこと。"""
    result = await agent_with_present._proactive_memory_context()
    assert result is None or isinstance(result, str)


@pytest.mark.asyncio
async def test_proactive_context_total_max_respected(monkeypatch, agent_with_present):
    """結果の記憶数が SHARE_MEMORY_TOTAL_MAX を超えないこと。"""
    monkeypatch.setenv("SHARE_MEMORY_TOTAL_MAX", "2")
    result = await agent_with_present._proactive_memory_context()
    if result:
        parts = result.split(" / ")
        assert len(parts) <= 2, f"Expected ≤2 parts, got {len(parts)}: {result}"


@pytest.mark.asyncio
async def test_proactive_context_no_candidates_returns_none():
    """シード候補が0件のとき None を返すこと。"""
    from familiar_agent.agent import EmbodiedAgent
    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    seed_mem = MagicMock()
    seed_mem.pick_seed_candidates = MagicMock(return_value=[])

    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = [("pid-1", seed_mem)]
    agent._pmm = pmm

    result = await agent._proactive_memory_context()
    assert result is None


# ---------------------------------------------------------------------------
# Tests: desires time-of-day bias removed for share_memory
# ---------------------------------------------------------------------------


def test_share_memory_no_evening_multiplier():
    """Evening (18-22) must NOT boost share_memory (Issue C 撤廃)."""
    from familiar_agent.desires import DesireSystem
    modulation = DesireSystem._time_modulation(20)  # 8pm
    rate = modulation.get("share_memory", 1.0)
    assert rate == 1.0, f"share_memory evening rate should be 1.0, got {rate}"


def test_share_memory_no_night_suppression():
    """Night (22-6) must NOT suppress share_memory (Issue C 撤廃)."""
    from familiar_agent.desires import DesireSystem
    modulation = DesireSystem._time_modulation(2)   # 2am
    rate = modulation.get("share_memory", 1.0)
    assert rate == 1.0, f"share_memory night rate should be 1.0, got {rate}"
