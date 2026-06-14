"""Tests for pending_speech priority in _proactive_memory_context (Issue D).

pending があれば Issue C 想起より優先して発話コンテンツが作られる。
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.config import PendingSpeechConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_row(obs_id: str = "o1", target: str | None = None,
               content: str = "話したい内容") -> dict:
    return {
        "id": f"ps-{obs_id}",
        "observation_id": obs_id,
        "target_person_id": target,
        "content": content,
        "created_at": datetime.now(timezone.utc),
        "reinforce_count": 0,
        "superseded_by": None,
    }


def _make_agent(pending_rows: list[dict] | None = None,
                present_ids: list[str] | None = None) -> EmbodiedAgent:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    pids = present_ids or ["pid-1"]

    pmm = MagicMock()
    mem_mocks = [(pid, MagicMock()) for pid in pids]
    for _pid, mem in mem_mocks:
        mem.pick_seed_candidates = MagicMock(return_value=[])
    pmm.get_all_present_memories.return_value = mem_mocks
    pmm.get_present_ids.return_value = pids
    pmm.get_person_name.side_effect = lambda pid: pid
    agent._pmm = pmm

    store = MagicMock()
    store.list_active.return_value = pending_rows or []
    store.freshness_score.return_value = 0.9
    store.is_expired.return_value = False
    store.delete = MagicMock()
    agent._pending_store = store

    persons = MagicMock()
    persons._trackers = {}
    agent._persons = persons

    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_spoken_before_recall(monkeypatch):
    """pending があれば Issue C 想起より優先して話される。"""
    monkeypatch.setenv("PENDING_SPEECH_MAX", "2")
    rows = [_fresh_row("o1", content="覚えた思い出")]
    agent = _make_agent(pending_rows=rows)
    result = await agent._proactive_memory_context()
    assert result is not None
    assert "覚えた思い出" in result


@pytest.mark.asyncio
async def test_pending_capped_at_max_per_turn(monkeypatch):
    """1回の発話で max_per_turn(デフォルト2)まで。"""
    monkeypatch.setenv("PENDING_SPEECH_MAX", "2")
    rows = [_fresh_row(f"o{i}", content=f"思い出{i}") for i in range(5)]
    agent = _make_agent(pending_rows=rows)
    result = await agent._proactive_memory_context()
    if result:
        parts = result.split(" / ")
        assert len(parts) <= 2


@pytest.mark.asyncio
async def test_null_pending_consumed_after_spoken(monkeypatch):
    """target=NULL は話したら削除される。"""
    monkeypatch.setenv("PENDING_SPEECH_MAX", "2")
    rows = [_fresh_row("o1", target=None, content="誰にでも話したい")]
    agent = _make_agent(pending_rows=rows)
    await agent._proactive_memory_context()
    agent._pending_store.delete.assert_called()


@pytest.mark.asyncio
async def test_expired_pending_removed_not_spoken(monkeypatch):
    """失効した pending は削除されるが発話に含まれない。"""
    monkeypatch.setenv("PENDING_SPEECH_MAX", "2")
    rows = [_fresh_row("o1", content="古すぎる記憶")]
    agent = _make_agent(pending_rows=rows)
    agent._pending_store.is_expired.return_value = True

    result = await agent._proactive_memory_context()
    # Expired row should be deleted
    agent._pending_store.delete.assert_called()
    # But should not appear in output (no pending alive → falls back)
    if result:
        assert "古すぎる記憶" not in result


@pytest.mark.asyncio
async def test_fallback_to_recall_when_no_pending():
    """pending 無し → Issue C 想起にフォールバック(None でも良い)。"""
    agent = _make_agent(pending_rows=[])
    # Make seed candidates empty to guarantee no recall result
    for _pid, mem in agent._pmm.get_all_present_memories.return_value:
        mem.pick_seed_candidates = MagicMock(return_value=[])
    result = await agent._proactive_memory_context()
    # No pending, no seeds → None
    assert result is None


@pytest.mark.asyncio
async def test_no_present_returns_none():
    """誰もいないとき None。"""
    agent = _make_agent(pending_rows=[], present_ids=[])
    agent._pmm.get_all_present_memories.return_value = []
    result = await agent._proactive_memory_context()
    assert result is None
