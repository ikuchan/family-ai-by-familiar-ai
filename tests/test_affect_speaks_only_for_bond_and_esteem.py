"""情動が話しかけるのは bond と esteem だけで、その 2 つはカメラに人が映っているときだけ溜まる
（出-as 段 8・2026-09-26・`設計方針_話していいかの決まり` §2.1）。

- bond・esteem：映っていれば溜まる（静穏時間は溜まる速さを 0.083 倍・いまの倍率）。映っていなければ、
  **溜まる速さの 1/3 の速さで減る**（本人の決定）。
- seeking・safety・rest：いまのまま（映っているかに関わらず溜まる）。
- 会話をしようとするのは bond・esteem が発火したときだけ。ほかの発火では、見る・首を振るなどの行動はするが、
  返事の文は声に出さない（声にしなかっただけの独り言として残す）。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core import drive_dynamics as dd
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

HALF = AiDrivers(seeking=0.5, rest=0.5, bond=0.5, safety=0.5, esteem=0.5)


def _mood():
    from familiar_agent.mood_register import MoodPAD

    return MoodPAD(p=0.2, pn=0.1, a=0.5, dom=0.5)


def _grow(visible: bool) -> AiDrivers:
    return dd.accumulate(HALF, _mood(), dt=10.0, cfg=DriveConfig(), someone_visible=visible)


def test_bond_and_esteem_grow_while_someone_is_visible():
    seen = _grow(True)
    assert seen.bond > 0.5 and seen.esteem > 0.5


def test_bond_and_esteem_fade_at_a_third_while_nobody_is_visible():
    seen, alone = _grow(True), _grow(False)
    for axis in ("bond", "esteem"):
        up = getattr(seen, axis) - 0.5
        down = 0.5 - getattr(alone, axis)
        assert down == pytest.approx(up / 3, rel=1e-6)


def test_the_other_axes_do_not_care_who_is_visible():
    seen, alone = _grow(True), _grow(False)
    for axis in ("seeking", "safety", "rest"):
        assert getattr(alone, axis) == pytest.approx(getattr(seen, axis))


def test_the_tonic_passes_what_the_camera_sees():
    from familiar_agent.loop import tonic

    assert "someone_visible" in inspect.signature(tonic.step_drives).parameters
    assert "someone_visible=self._someone_visible()" in inspect.getsource(tonic.Tonic._run)


def test_without_a_camera_nobody_is_visible():
    from familiar_agent.loop.tonic import Tonic

    t = Tonic(MagicMock(), presence=None)
    assert t._someone_visible() is False
    sensor = MagicMock()
    sensor.room_occupied = MagicMock(return_value=True)
    assert Tonic(MagicMock(), presence=sensor)._someone_visible() is True


# ── 話しかけるのは bond・esteem だけ ─────────────────────────────────────


def _affect(axis: str):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "情動"
    ip._req.fired_axis = axis
    ip._delivery_block_reason = lambda: ""
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()
    return ip


@pytest.mark.parametrize("axis", ["seeking", "safety", "rest"])
def test_other_urges_do_not_talk(axis):
    ip = _affect(axis)
    spoken, outcome = asyncio.run(ip._speak("見てみよう"))
    assert outcome == "沈黙"
    ip._dif.speak.assert_not_awaited()


@pytest.mark.parametrize("axis", ["bond", "esteem"])
def test_bond_and_esteem_talk(axis):
    ip = _affect(axis)
    spoken, outcome = asyncio.run(ip._speak("ねえ、元気？"))
    assert outcome == "発話"
    ip._dif.speak.assert_awaited_once()
