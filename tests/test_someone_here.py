"""「居る」の合成（知-ai 段 3・2026-10-05・本人の決定・`設計方針_在席と顔ぶれ` v0.1）。

**居る**＝誰かの顔ぶれの持ち時間が残っている（特定の誰かがいる）、または、全員切れていれば在席（カメラが人を
見ている・不特定の誰か）。顔ぶれが先で、在席は 2 番目。首を回して写らなくなっても、顔ぶれの持ち時間のあいだは
居るとみなす。

これを見るのは 4 か所：出口の門（情動と機器の知らせ）・BOND と ESTEEM の溜まり方・REST 内省に入るか・
窓の外の声で見に行く理由。以前はどれも在席だけを見ていた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _agent_with(*, present: list, occupied: bool):
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._pmm = MagicMock()
    a._pmm.present_keys = MagicMock(return_value=list(present))
    a._occupancy = MagicMock(return_value=1.0 if occupied else 0.0)
    return a


@pytest.mark.parametrize(
    "present,occupied,want",
    [
        (["papa"], False, True),  # 顔ぶれだけ（首を回して写らない）
        ([], True, True),  # 在席だけ（誰かは分からない）
        (["unknown:1"], False, True),  # 名前の分からない顔ぶれも居る
        ([], False, False),
    ],
)
def test_someone_here_is_presence_first_then_occupancy(present, occupied, want):
    from familiar_agent.agent import EmbodiedAgent

    a = _agent_with(present=present, occupied=occupied)
    assert EmbodiedAgent._someone_here(a) is want


# ── 出口の門 ─────────────────────────────────────────────────────────────────


def _ip(*, here: bool):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._someone_here = MagicMock(return_value=here)
    a._occupancy = MagicMock(return_value=0.0)  # カメラは誰も見ていない
    ip = InformationProcessing(a)
    return ip, a


def test_the_exit_gate_lets_affect_speak_while_someone_is_here():
    ip, _ = _ip(here=True)
    ip._req.trigger_kind = "情動"
    assert ip._delivery_block_reason() == ""
    ip, _ = _ip(here=False)
    ip._req.trigger_kind = "情動"
    assert ip._delivery_block_reason() == "聞く相手が居ない"


@pytest.mark.real_window
def test_a_dropped_voice_nudges_only_while_someone_is_here():
    """情-p：誰か居るときだけ押し上げる（以前は誰も居ないときだけ SEEKING を押し上げた）。"""
    from familiar_agent.loop.event_loop import Trigger

    for here, nudged in ((True, True), (False, False)):
        ip, a = _ip(here=here)
        a._nudge_drive = AsyncMock()
        a.config.agent_names = ["パジュ"]
        ip._load_silence = lambda: None  # type: ignore[method-assign]
        fut = asyncio.new_event_loop().create_future()
        t = Trigger(kind="会話入力", query="ごはんまだ？", future=fut, source="voice", arrived=0.0)
        asyncio.run(ip._swallow_if_unheard(t))
        assert a._nudge_drive.await_count == (1 if nudged else 0), here


# ── T：欲求の溜まり方と REST ─────────────────────────────────────────────────


def _tonic(*, present: list, occupied: "bool | None"):
    from familiar_agent.loop.tonic import Tonic

    agent = MagicMock()
    agent._pmm.present_keys = MagicMock(return_value=list(present))
    sensor = None
    if occupied is not None:
        sensor = MagicMock()
        sensor.room_occupied = MagicMock(return_value=occupied)
    return Tonic(MagicMock(), agent=agent, occupancy=sensor)


def test_social_drives_grow_while_someone_is_here():
    assert _tonic(present=["papa"], occupied=False)._someone_here() is True
    assert _tonic(present=[], occupied=True)._someone_here() is True
    assert _tonic(present=[], occupied=False)._someone_here() is False
    assert _tonic(present=[], occupied=None)._someone_here() is False  # カメラの無い機体


def test_the_tick_passes_someone_here_to_the_drives():
    import inspect

    from familiar_agent.loop import tonic

    assert "occupied=self._someone_here()" in inspect.getsource(tonic.Tonic._run)
