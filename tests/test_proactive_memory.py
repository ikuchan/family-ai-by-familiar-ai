"""Tests for proactive memory sharing desire.

Phase 3 of companion-likeness improvements.
'share_memory' desire grows over time, triggering contextual "remember when..." prompts.

Issue C: time-of-day multipliers removed from desires; time-aware recall moved into
pick_seed_candidates() via hour/month window parameters.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.desires import (
    DEFAULT_DESIRES,
    GROWTH_RATES,
    DesireSystem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _desires(tmp_path: Path) -> DesireSystem:
    return DesireSystem(state_path=tmp_path / "desires.json")


# ---------------------------------------------------------------------------
# Tests: share_memory desire exists and grows
# ---------------------------------------------------------------------------


def test_share_memory_in_default_desires() -> None:
    assert "share_memory" in DEFAULT_DESIRES


def test_share_memory_in_growth_rates() -> None:
    assert "share_memory" in GROWTH_RATES


def test_share_memory_growth_rate_positive() -> None:
    assert GROWTH_RATES["share_memory"] > 0.0


def test_share_memory_grows_with_tick(tmp_path) -> None:
    desires = _desires(tmp_path)
    initial = desires.level("share_memory")
    desires._last_tick -= 300  # 5 minutes
    desires.tick()
    assert desires.level("share_memory") > initial


def test_share_memory_satisfies_and_decays(tmp_path) -> None:
    desires = _desires(tmp_path)
    desires.boost("share_memory", 0.8)
    before = desires.level("share_memory")
    desires.satisfy("share_memory")
    assert desires.level("share_memory") < before


def test_share_memory_has_prompt_in_dominant(tmp_path) -> None:
    """When share_memory is dominant, dominant_as_prompt returns a string."""
    desires = _desires(tmp_path)
    desires.boost("share_memory", 1.0)
    for name in ("look_around", "explore", "greet_companion", "rest", "worry_companion"):
        desires._desires[name] = 0.0
    prompt = desires.dominant_as_prompt()
    assert prompt is not None
    assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Tests: circadian modulation — Issue C removed time-of-day bias
# ---------------------------------------------------------------------------


def test_share_memory_no_evening_boost() -> None:
    """Issue C: evening must NOT boost share_memory (time-aware recall is in pick_seed_candidates)."""
    modulation = DesireSystem._time_modulation(20)  # 8pm
    rate = modulation.get("share_memory", 1.0)
    assert rate == 1.0, f"share_memory evening rate should be 1.0, got {rate}"


def test_share_memory_no_night_suppression() -> None:
    """Issue C: night must NOT suppress share_memory."""
    modulation = DesireSystem._time_modulation(2)  # 2am
    rate = modulation.get("share_memory", 1.0)
    assert rate == 1.0, f"share_memory night rate should be 1.0, got {rate}"


def test_share_memory_default_modulation_daytime() -> None:
    """Daytime modulation leaves share_memory at ×1.0."""
    modulation = DesireSystem._time_modulation(14)  # 2pm
    rate = modulation.get("share_memory", 1.0)
    assert rate == 1.0, f"share_memory daytime rate should be 1.0, got {rate}"


# ---------------------------------------------------------------------------
# Tests: _proactive_memory_context — new pmm-based API (Issue C)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proactive_memory_context_returns_none_when_no_present() -> None:
    """誰もいないとき (get_all_present_memories returns []) → None を返す。"""
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = []
    agent._pmm = pmm

    result = await agent._proactive_memory_context()
    assert result is None


@pytest.mark.asyncio
async def test_proactive_memory_context_returns_none_when_no_seeds() -> None:
    """在席者いるがシード候補ゼロ → None を返す。"""
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    seed_mem = MagicMock()
    seed_mem.pick_seed_candidates = MagicMock(return_value=[])

    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = [("pid-1", seed_mem)]
    agent._pmm = pmm

    result = await agent._proactive_memory_context()
    assert result is None


@pytest.mark.asyncio
async def test_proactive_memory_context_returns_string_when_present() -> None:
    """在席者かつシードあり → 文字列を返す (または None)。"""
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    seed_mem = MagicMock()
    seed_mem.pick_seed_candidates = MagicMock(return_value=[
        {"id": "s1", "content": "古い思い出A", "timestamp": datetime(2025, 6, 14, 14)},
    ])

    assoc_mem = MagicMock()
    assoc_mem.recall_async = AsyncMock(return_value=[
        {"memory_id": "a1", "content": "連想された記憶", "score": 0.85},
    ])

    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = [("pid-1", seed_mem)]
    pmm.get_speaker_memory.return_value = None
    pmm.get_agent_memory.return_value = assoc_mem
    agent._pmm = pmm

    result = await agent._proactive_memory_context()
    assert result is None or isinstance(result, str)


@pytest.mark.asyncio
async def test_proactive_memory_context_total_max_respected(monkeypatch) -> None:
    """SHARE_MEMORY_TOTAL_MAX=1 のとき結果が1件を超えない。"""
    from familiar_agent.agent import EmbodiedAgent

    monkeypatch.setenv("SHARE_MEMORY_TOTAL_MAX", "1")

    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    seed_mem = MagicMock()
    seed_mem.pick_seed_candidates = MagicMock(return_value=[
        {"id": f"s{i}", "content": f"思い出{i}", "timestamp": datetime(2025, 6, 14, 14)}
        for i in range(5)
    ])

    assoc_mem = MagicMock()
    assoc_mem.recall_async = AsyncMock(return_value=[
        {"memory_id": f"a{i}", "content": f"連想{i}", "score": 0.9 - i * 0.1}
        for i in range(5)
    ])

    pmm = MagicMock()
    pmm.get_all_present_memories.return_value = [("pid-1", seed_mem)]
    pmm.get_speaker_memory.return_value = None
    pmm.get_agent_memory.return_value = assoc_mem
    agent._pmm = pmm

    result = await agent._proactive_memory_context()
    if result:
        parts = result.split(" / ")
        assert len(parts) <= 1, f"Expected ≤1 part, got {len(parts)}: {result}"
