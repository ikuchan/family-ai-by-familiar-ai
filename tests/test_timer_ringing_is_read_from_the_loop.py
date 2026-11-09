"""タイマーの音が鳴っているかは、ループの DIF から読む（2026-11-09・本人の承認）。

DIF はループ（`agent._info_processing._dif`）が持ち、`agent` には `_dif` が無い。`_timer_active`（Jev の操作の言葉の守り・
出-aa）と画面のタイマーの欄が `agent._dif` を読んでいたので、鳴っていても「鳴っていない」と読んでいた。前にも同じ
取り違えで、メモの見回りが実機で落ちた（c8a6a1c1）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _agent(ringing: "bool | None"):
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._timer_tool = None
    a._alarm_tool = None
    a._stopwatch_tool = None
    if ringing is not None:
        a._info_processing = SimpleNamespace(_dif=SimpleNamespace(ringing=ringing))
    return a


@pytest.mark.parametrize("ringing", [True, False])
def test_the_agent_says_whether_the_timer_rings(ringing):
    assert _agent(ringing).timer_ringing() is ringing


def test_without_a_loop_nothing_rings():
    """**反証の側**：ループがまだ無ければ、鳴っていない。"""
    assert _agent(None).timer_ringing() is False


@pytest.mark.parametrize("ringing", [True, False])
def test_the_gate_sees_the_ring_without_a_timer_frame(ringing):
    """タイマーの欄が無くても、音が鳴っていれば操作の言葉を守る（出-aa）。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._agent = _agent(None)  # agent 側には DIF が無い
    ip._dif = SimpleNamespace(ringing=ringing)
    assert ip._timer_active() is ringing


@pytest.mark.parametrize("ringing", [True, False])
def test_the_panel_shows_the_ring(ringing):
    from familiar_agent.gui import PresencePanel

    panel = PresencePanel.__new__(PresencePanel)
    agent = _agent(ringing)
    panel._get_agent = lambda: agent
    rows = panel._timer_rows()
    assert ("🔔 鳴っている" in rows) is ringing
