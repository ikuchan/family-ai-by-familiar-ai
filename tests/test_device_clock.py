"""端-c c6a：端末の求めのタイマーとアラームは、頼んだ端末の時計アプリで掛ける（課題9 b2・§8.8）。

- 機体は家のタイマーとアラームに入れない（家で鳴らない・家の確認も黙りも置かない）。端末へ `timer`（秒と名前）か
  `alarm`（時と分と名前）を送り、端末が時計アプリへ渡す。
- 機体は端末の時計に掛かったかを知らないので、返りは「端末の時計に頼んだ」と書く（掛けた、とは書かない）。
- 止める・一時停止・ストップウォッチ・音楽を止める道具・印刷・確認の答えは、端末の求めではいままでどおり使わない。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.device_clock import TIMER_MAX_SEC, clock_order
from familiar_agent.loop.event_loop import InformationProcessing

pytestmark = pytest.mark.real_window

TAB = "tablet-1"
NOW = datetime(2026, 11, 26, 6, 30, 10, tzinfo=timezone(timedelta(hours=9)))


def test_a_timer_becomes_seconds_and_a_name():
    assert clock_order("set_timer", {"after_minutes": 3, "label": "パスタ"}, now=NOW) == (
        "timer",
        {"seconds": 180, "label": "パスタ"},
    )
    assert clock_order("set_timer", {"after_minutes": 0.5, "label": ""}, now=NOW) == (
        "timer",
        {"seconds": 30, "label": "タイマー"},
    )


def test_an_alarm_becomes_the_hour_and_the_minute():
    assert clock_order("set_alarm", {"at": "7時半", "label": "起こす"}, now=NOW) == (
        "alarm",
        {"hour": 7, "minute": 30, "label": "起こす"},
    )
    # 過ぎた時刻でも時と分だけを送る（翌日に鳴らすのは時計アプリ）
    assert clock_order("set_alarm", {"at": "6:00", "label": "薬"}, now=NOW) == (
        "alarm",
        {"hour": 6, "minute": 0, "label": "薬"},
    )


def test_what_the_clock_cannot_take_is_told_back():
    for action, inp in (
        ("set_timer", {"after_minutes": 0, "label": "x"}),
        ("set_timer", {"label": "x"}),
        ("set_timer", {"after_minutes": 1, "at": "7:00", "label": "x"}),
        ("set_timer", {"after_minutes": TIMER_MAX_SEC / 60 + 1, "label": "x"}),
        ("set_timer", {"after_minutes": 0.001, "label": "x"}),  # 1 秒に満たない
        ("set_timer", {"after_minutes": 1e12, "label": "x"}),  # 時刻にできないほど長い
        ("set_alarm", {"at": "25:00", "label": "x"}),
        ("set_alarm", {"at": "あした", "label": "x"}),
    ):
        got = clock_order(action, inp, now=NOW)
        assert isinstance(got, str) and got, (action, inp)
    assert (
        TIMER_MAX_SEC == 86_400
    )  # 時計アプリが受ける長さ（Android の AlarmClock.EXTRA_LENGTH は 1〜86400 秒）


def _ip(device: str = TAB):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a.config.tts_gain = 1.0
    a._pmm.presence_status = MagicMock(return_value=[])
    a._timer_tool.call = AsyncMock(return_value=("掛けた：id=1", True))
    a._alarm_tool.call = AsyncMock(return_value=("掛けた：id=1", True))
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    ip._req.trigger_kind = "発話"
    ip._req.device = device
    got: list = []
    ip.set_device_listener(lambda dev, kind, payload: got.append((dev, kind, payload)))
    return ip, got


def _run(ip, action, inp):
    asyncio.run(ip._run_lookup_body(action, inp, "q", None))
    return ip._triggers.get_nowait()


def test_a_device_timer_is_sent_to_the_device_and_not_set_at_home():
    ip, got = _ip()
    done = _run(ip, "set_timer", {"after_minutes": 3, "label": "パスタ"})
    assert got == [(TAB, "timer", {"seconds": 180, "label": "パスタ"})]
    ip._agent._timer_tool.call.assert_not_called()
    assert done.kind == "完了" and not done.failed
    assert "端末の時計" in done.result and "頼んだ" in done.result


def test_a_device_alarm_is_sent_to_the_device_and_not_set_at_home():
    ip, got = _ip()
    done = _run(ip, "set_alarm", {"at": "7:00", "label": "起こす"})
    assert got == [(TAB, "alarm", {"hour": 7, "minute": 0, "label": "起こす"})]
    ip._agent._alarm_tool.call.assert_not_called()
    assert not done.failed and "端末の時計" in done.result


def test_a_device_timer_is_sent_only_once_per_request():
    """家のタイマーは「同時に 1 本」で掛け直しを止める。端末の時計は機体から見えないので、1 つの求めで 1 回までにする。"""
    ip, got = _ip()
    _run(ip, "set_timer", {"after_minutes": 3, "label": "パスタ"})
    again = _run(ip, "set_timer", {"after_minutes": 3, "label": "パスタ"})
    assert len(got) == 1 and again.failed
    assert "set_timer" not in set(
        ip._extra_actions()
    )  # 使い切った動作は、次の反復の調停にも出さない


def test_the_length_is_told_in_minutes_and_seconds():
    ip, _ = _ip()
    done = _run(ip, "set_timer", {"after_minutes": 100 / 60, "label": "お茶"})
    assert "1 分 40 秒の「お茶」" in done.result


def test_what_the_device_clock_cannot_take_fails_without_sending():
    ip, got = _ip()
    done = _run(ip, "set_timer", {"after_minutes": 0, "label": "x"})
    assert got == [] and done.failed


def test_a_home_timer_is_still_set_at_home():
    """**反証の側**：家の求めは、いままでどおり家のタイマーで掛け、端末へは送らない。"""
    ip, got = _ip(device="")
    _run(ip, "set_timer", {"after_minutes": 3, "label": "パスタ"})
    ip._agent._timer_tool.call.assert_awaited_once()
    assert got == []


@pytest.mark.parametrize(
    "action", ["cancel_timer", "pause_timer", "cancel_alarm", "start_stopwatch", "stop_music"]
)
def test_other_sound_tools_are_still_refused_for_a_device(action):
    ip, got = _ip()
    done = _run(ip, action, {"id": "all"})
    assert done.failed and got == []
    ip._agent._timer_tool.call.assert_not_called()


def test_the_device_is_offered_the_timer_and_the_alarm():
    ip, _ = _ip()
    ip._gated = lambda defs: defs or [{"name": "x"}]
    offered = set(ip._extra_actions())
    assert {"set_timer", "set_alarm"} <= offered
    assert not offered & {"cancel_timer", "cancel_alarm", "start_stopwatch", "confirm"}
