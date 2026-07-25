"""Tests for social-presence gating on desire firing."""

from __future__ import annotations

import time
from unittest.mock import MagicMock


from familiar_agent.desires import DesireSystem


def _make_desires() -> DesireSystem:
    return DesireSystem(companion_name="Taro")


# ── social_permission via update_context ──────────────────────────────────────

def test_social_desire_blocked_when_permission_zero():
    desires = _make_desires()
    # Force greet_companion to max level
    desires._desires["greet_companion"] = 1.0
    # Block social
    desires.update_context(social_permission=0.0)
    result = desires.get_dominant()
    # greet_companion should be blocked; some internal desire may or may not fire
    if result is not None:
        from familiar_agent.desires import is_social_desire
        assert not is_social_desire(result[0]), f"Social desire {result[0]} fired despite permission=0"


def test_social_desire_allowed_when_permission_one():
    desires = _make_desires()
    desires._desires["greet_companion"] = 1.0
    desires.update_context(social_permission=1.0)
    result = desires.get_dominant()
    assert result is not None
    assert result[0] == "greet_companion"


def test_internal_desire_unaffected_by_permission_zero():
    desires = _make_desires()
    desires._desires["look_around"] = 1.0
    # Set all social desires to 0 to ensure look_around wins
    for name in ["greet_companion", "share_memory", "attachment", "care", "repair", "play"]:
        desires._desires[name] = 0.0
    desires.update_context(social_permission=0.0)
    result = desires.get_dominant()
    assert result is not None
    assert result[0] == "look_around"


# ── _social_presence_permission logic (via mock agent) ────────────────────────

class _MinimalAgent:
    """Minimal stub that exposes only the attributes _social_presence_permission needs."""
    def __init__(self):
        self._last_human_at: float = time.time()
        self._presence_watcher = None
        self._pmm = None

    # Copy the method under test directly
    _social_presence_permission = None  # replaced below


# Attach the real method from agent.py
from familiar_agent.agent import EmbodiedAgent
_MinimalAgent._social_presence_permission = EmbodiedAgent._social_presence_permission


def _agent_stub(**kwargs) -> _MinimalAgent:
    stub = _MinimalAgent()
    for k, v in kwargs.items():
        setattr(stub, k, v)
    return stub


def test_no_camera_recent_interaction_permits_social():
    stub = _agent_stub(_last_human_at=time.time() - 30)  # 30s ago
    assert stub._social_presence_permission() == 1.0


def test_no_camera_old_interaction_blocks_social():
    stub = _agent_stub(_last_human_at=time.time() - 400)  # 400s ago (> 5 min)
    assert stub._social_presence_permission() == 0.0


def test_no_camera_exactly_at_boundary():
    stub = _agent_stub(_last_human_at=time.time() - 299)  # just under 5 min
    assert stub._social_presence_permission() == 1.0


def test_no_camera_no_last_human_at_blocks_social():
    stub = _MinimalAgent()
    del stub._last_human_at
    assert stub._social_presence_permission() == 0.0


def test_camera_active_person_present_permits_social():
    pmm = MagicMock()
    pmm.get_present_ids.return_value = ["person_1"]
    stub = _agent_stub(_presence_watcher=MagicMock(), _pmm=pmm)
    assert stub._social_presence_permission() == 1.0


def test_camera_active_no_person_but_recent_utterance_permits_social():
    # 顔が見えなくても、話しかけられていれば人は居る（在席の証拠は顔と対話の二つで、
    # どちらかが立てば在席）。以前は顔だけを見て確定し、目の前の相手への返事まで
    # 保留にしていた。identity と presence の分離は残課題 #8。
    pmm = MagicMock()
    pmm.get_present_ids.return_value = []
    stub = _agent_stub(_presence_watcher=MagicMock(), _pmm=pmm,
                       _last_human_at=time.time() - 30)
    assert stub._social_presence_permission() == 1.0


def test_camera_active_no_person_and_no_utterance_blocks_social():
    pmm = MagicMock()
    pmm.get_present_ids.return_value = []
    stub = _agent_stub(_presence_watcher=MagicMock(), _pmm=pmm,
                       _last_human_at=time.time() - 600)   # 10分前＝在席の証拠なし
    assert stub._social_presence_permission() == 0.0


def test_camera_active_no_pmm_falls_back_to_the_utterance():
    stub = _agent_stub(_presence_watcher=MagicMock(), _pmm=None,
                       _last_human_at=time.time() - 600)
    assert stub._social_presence_permission() == 0.0
