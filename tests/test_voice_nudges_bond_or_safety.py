"""窓の外で捨てた声で、知っている声なら BOND、知らない声なら SAFETY を押し上げる（情-p・2026-10-10・本人の決定）。

以前は、窓の外の声を捨てるたびに、誰も見えなければ SEEKING へ発火閾値の半分（0.50）を足していた（案ア・2026-09-17）。
上限が無く、seeking の動作は検索だけなので、家族どうしが話しているあいだ声 2 つごとに天気とサッカーを検索した
（10/10 の SEEKING からの検索 68 回・12 時台 29 回）。押し上げは、**カメラに誰か居るとき**だけにし、声で分ける：

| 誰か居るか | 声 | すること |
|---|---|---|
| 居る | 家族のだれかに当たる（`_voice_unknown` が偽・0.35 以上） | BOND +0.05（声 20 回に 1 回ほど） |
| 居る | 誰にも当たらない・声の特徴が無い | SAFETY +0.10（声 10 回に 1 回ほど） |
| 居ない | どちらでも | 何もしない |

押し上げるのは、声で届いて窓の外として捨てた入力だけ（キーボード・窓の中・黙っている間は押し上げない）。発火は通常の
tick が決める。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core import drive_dynamics as dd
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.event_loop import InformationProcessing, Trigger

pytestmark = pytest.mark.real_window  # 門そのものを確かめる（conftest の窓の開き口を使わない）

KNOWN = "知っている声"
STRANGER = "知らない声"


def test_nudge_adds_to_one_axis_and_stops_at_the_threshold():
    cfg = DriveConfig()
    d = AiDrivers(seeking=0.3, bond=0.2)
    out = dd.nudge(d, "bond", 0.5, cfg)
    assert abs(out.bond - 0.7) < 1e-9 and out.seeking == 0.3
    assert dd.nudge(out, "bond", 0.5, cfg).bond == cfg.theta_fire


def test_the_default_amounts():
    with patch.dict(os.environ, {}, clear=True):
        cfg = DriveConfig()
    assert (cfg.voice_bond_nudge, cfg.voice_safety_nudge) == (0.05, 0.10)


def _ip(*, here: bool):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a._pmm.presence_status = MagicMock(return_value=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    a._occupancy = MagicMock(return_value=1.0 if here else 0.0)
    a._someone_here = MagicMock(return_value=here)
    a._nudge_drive = AsyncMock()
    a._timer_tool.frame = MagicMock(return_value="")
    a._dif.ringing = False
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None  # type: ignore[method-assign]  # 黙ってはいない
    ip._voice_unknown = lambda v: v != KNOWN  # type: ignore[method-assign]
    return ip, a


def _drop(ip, text="ごはんまだ？", *, voice=KNOWN, source="voice"):
    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(kind="会話入力", query=text, future=fut, source=source, voice=voice)
        return await ip._swallow_if_unheard(t)

    return asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def _nudged(a):
    return [(c.args[0], c.args[1]) for c in a._nudge_drive.await_args_list]


def test_a_known_voice_while_someone_is_here_nudges_bond():
    ip, a = _ip(here=True)
    assert _drop(ip, voice=KNOWN) is True
    assert _nudged(a) == [("bond", pytest.approx(0.05))]


@pytest.mark.parametrize("voice", [STRANGER, None])
def test_a_stranger_while_someone_is_here_nudges_safety(voice):
    ip, a = _ip(here=True)
    assert _drop(ip, voice=voice) is True
    assert _nudged(a) == [("safety", pytest.approx(0.10))]


@pytest.mark.parametrize("voice", [KNOWN, STRANGER, None])
def test_nothing_while_nobody_is_here(voice):
    ip, a = _ip(here=False)
    assert _drop(ip, voice=voice) is True
    assert _nudged(a) == []


def test_typing_does_not_nudge():
    ip, a = _ip(here=True)
    assert _drop(ip, voice=None, source="keyboard") is True
    assert _nudged(a) == []


def test_a_heard_voice_does_not_nudge():
    ip, a = _ip(here=True)
    assert _drop(ip, "パジュ、おはよう") is False  # 名前で窓が開いて受けた
    assert _drop(ip, "明日の天気は？") is False  # 窓の中
    assert _nudged(a) == []


def test_a_voice_while_silenced_does_not_nudge():
    from familiar_agent.silence_state import SilenceRequest

    ip, a = _ip(here=True)
    import time

    ip._load_silence = lambda: SilenceRequest(person="パパ", until=time.time() + 600)  # type: ignore[method-assign]
    ip._note_muted = AsyncMock()  # type: ignore[method-assign]
    assert _drop(ip, "ごはんまだ？") is True
    ip._note_muted.assert_awaited_once()  # 黙っている間の道を通った
    assert _nudged(a) == []
