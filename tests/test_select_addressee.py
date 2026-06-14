"""Tests for _select_addressee (Issue D).

複数人がいる場面で誰に話しかけるかを、
話したい内容の強さと関係性から確率的に決める。
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.config import PendingSpeechConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_row(target: str | None = None, content: str = "x") -> dict:
    return {
        "id": str(__import__("uuid").uuid4()),
        "target_person_id": target,
        "content": content,
        "created_at": datetime.now(timezone.utc),
        "reinforce_count": 0,
        "superseded_by": None,
    }


def _make_agent(present_ids: list[str], rel_by_pid: dict | None = None) -> EmbodiedAgent:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    pmm = MagicMock()
    pmm.get_present_ids.return_value = present_ids
    pmm.get_person_name.side_effect = lambda pid: pid
    agent._pmm = pmm

    persons = MagicMock()
    trackers: dict[str, MagicMock] = {}
    if rel_by_pid:
        for pid, (trust, intimacy) in rel_by_pid.items():
            t = MagicMock()
            t.trust = trust
            t.intimacy = intimacy
            trackers[pid] = t
    persons._trackers = trackers
    agent._persons = persons

    # PendingSpeechStore mock (not used in _select_addressee directly)
    agent._pending_store = MagicMock()

    return agent


@pytest.fixture()
def cfg():
    return PendingSpeechConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_none_when_no_present(cfg):
    """present が空なら None。"""
    agent = _make_agent([])
    assert agent._select_addressee([], [], cfg) is None


def test_returns_only_present_when_one_person(cfg):
    """1人しかいなければその人に決まる。"""
    agent = _make_agent(["pid-1"])
    rows = [_fresh_row(target="pid-1")]
    result = agent._select_addressee(["pid-1"], rows, cfg)
    assert result == "pid-1"


def test_null_pending_boosts_all_present(cfg):
    """target=NULL は全員の内容の強さを底上げする（両者に非ゼロ確率）。"""
    agent = _make_agent(["pid-1", "pid-2"])
    rows = [_fresh_row(target=None, content="誰にでも話したい")]
    results: set[str] = set()
    for _ in range(60):
        r = agent._select_addressee(["pid-1", "pid-2"], rows, cfg)
        if r:
            results.add(r)
    # Both should appear since NULL boosts everyone
    assert len(results) == 2


def test_temperature_default_is_one(cfg):
    """デフォルト temperature は 1.0（完全比例）。"""
    assert cfg.temperature == 1.0


def test_addressee_weighted_by_content(cfg):
    """ターゲット指定が多い相手が選ばれやすい（確率的だが統計的に優位）。"""
    agent = _make_agent(["pid-A", "pid-B"])
    # 5 rows targeting pid-A, 1 targeting pid-B
    rows = [_fresh_row(target="pid-A", content=f"msg{i}") for i in range(5)]
    rows += [_fresh_row(target="pid-B", content="msgB")]
    counts: dict[str, int] = {"pid-A": 0, "pid-B": 0}
    for _ in range(100):
        r = agent._select_addressee(["pid-A", "pid-B"], rows, cfg)
        if r:
            counts[r] = counts.get(r, 0) + 1
    assert counts["pid-A"] > counts["pid-B"]


def test_absent_person_not_selected(cfg):
    """present にいない person は選ばれない。"""
    agent = _make_agent(["pid-1"])
    rows = [_fresh_row(target="pid-99"), _fresh_row(target="pid-1")]
    for _ in range(20):
        r = agent._select_addressee(["pid-1"], rows, cfg)
        assert r != "pid-99"


def test_no_rows_returns_none_or_random(cfg):
    """pending が空でも present がいれば選ぶか None（クラッシュしない）。"""
    agent = _make_agent(["pid-1"])
    result = agent._select_addressee(["pid-1"], [], cfg)
    assert result is None or result == "pid-1"
