"""タイマーを掛けているあいだは黙り、鳴ったら元に戻す（`TIMER_SILENCE`・既定 有効・2026-09-16）。

「10秒のタイマーをかけてその間黙ってて」（実機 17:00）——掛けるのと黙るのを別々に頼まなくて
よいように、掛けた瞬間から鳴るまで黙る。既存の沈黙依頼と同じ仕組み（`SilenceRequest`）を使い、
**鳴る時刻＝期限**にする。鳴る知らせは何もしなくても通り、溜めていた返事も一緒に配る。
止めたら解く。明示の「黙って」は上書きしない。ストップウォッチは対象外。
`.env` の `TIMER_SILENCE=false` でこれまでどおり（掛けても黙らない）。
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from familiar_agent.config import AgentConfig
from familiar_agent.core.silence_rules import silence_note
from familiar_agent.loop import timer_watch
from familiar_agent.routines import QuietHoursRule
from familiar_agent.silence_state import SilenceRequest, hush_for_timer, unhush_timer
from familiar_agent.tools.timer import TimerTool

from tests.test_timer_tool import _FakeStore

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 9, 16, 17, 0, tzinfo=JST)


def test_the_switch_defaults_to_on_and_can_be_turned_off():
    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().timer_silence is True
    with patch.dict(os.environ, {"TIMER_SILENCE": "false"}, clear=True):
        assert AgentConfig().timer_silence is False


# ── 状態の規則（純関数：いまの依頼と、掛けたタイマーから次の依頼を決める）────────


def test_a_timer_hushes_until_it_rings():
    req = hush_for_timer(None, person="パパ", until=100.0, tid=6)
    assert req == SilenceRequest(person="パパ", until=100.0, reason="timer:6")


def test_an_explicit_request_is_not_overwritten():
    explicit = SilenceRequest(person="パパ", until=500.0, reason="")
    assert hush_for_timer(explicit, person="パパ", until=100.0, tid=6, now=0.0) is explicit


def test_an_expired_explicit_request_gives_way():
    stale = SilenceRequest(person="パパ", until=50.0, reason="")
    req = hush_for_timer(stale, person="パパ", until=100.0, tid=6, now=60.0)
    assert req.reason == "timer:6" and req.until == 100.0


def test_two_timers_hush_until_the_later_one():
    first = SilenceRequest(person="パパ", until=200.0, reason="timer:6")
    req = hush_for_timer(first, person="パパ", until=100.0, tid=7, now=0.0)
    assert req.until == 200.0 and req.reason == "timer:6"


def test_cancelling_the_timer_lifts_its_hush_but_not_an_explicit_one():
    assert unhush_timer(SilenceRequest("パパ", 100.0, reason="timer:6"), tid=6) is None
    kept = SilenceRequest("パパ", 100.0, reason="")
    assert unhush_timer(kept, tid=6) is kept
    other = SilenceRequest("パパ", 100.0, reason="timer:7")
    assert unhush_timer(other, tid=6) is other
    assert unhush_timer(other, tid="all") is None


def test_the_arbiter_note_says_it_is_a_timer():
    req = SilenceRequest("パパ", until=time.time() + 120, reason="timer:6")
    note = silence_note(req, now=time.time())
    assert "タイマーが鳴るまで黙っている" in note and "分" in note


# ── 道具からの呼び出し ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _confirm_off(monkeypatch):
    """ここは掛かったあとの振る舞いを見る。確かめる（`TIMER_CONFIRM`）は `test_timer_confirm_and_single` が見る。"""
    monkeypatch.setenv("TIMER_CONFIRM", "false")


def _tool():
    store = _FakeStore()
    oif = MagicMock()
    oif.write = AsyncMock(return_value="obs-1")
    hushed: list = []
    unhushed: list = []
    t = TimerTool(
        store=lambda: store,
        oif=oif,
        speaker=lambda: "パパ",
        quiet=lambda: QuietHoursRule(23, 7),
        silence_active=lambda: False,
        now=lambda: NOW,
        hush=lambda person, until, tid: hushed.append((person, until, tid)),
        unhush=lambda tid: unhushed.append(tid),
    )
    return t, hushed, unhushed


def test_setting_a_timer_hushes_until_due():
    t, hushed, _ = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and "黙って" in text
    assert hushed == [("パパ", NOW + timedelta(minutes=3), 1)]


def test_a_stopwatch_does_not_hush():
    t, hushed, _ = _tool()
    asyncio.run(t.call("start_stopwatch", {"label": "測る"}))
    assert hushed == []


def test_the_switch_off_keeps_the_old_behaviour(monkeypatch):
    monkeypatch.setenv("TIMER_SILENCE", "false")  # 設定は `.env` が正本・呼ぶたびに読む（知-o）
    t, hushed, _ = _tool()
    text, _ = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert hushed == [] and "黙って" not in text


def test_cancelling_unhushes():
    t, _, unhushed = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    asyncio.run(t.call("cancel_timer", {"id": 1}))
    assert unhushed == [1]
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "お茶"}))
    asyncio.run(t.call("cancel_timer", {"id": "all"}))
    assert unhushed == [1, "all"]


# ── 鳴った知らせは不在の保留を配らない（黙っていた分は聞いたことの列挙で載る・情-h）──


def test_a_ringing_timer_does_not_release_absence_speech():
    store = MagicMock()
    store.due_now = MagicMock(
        return_value=[
            {
                "id": 1,
                "label": "パスタ",
                "due": NOW - timedelta(seconds=1),
                "asked_by": "パパ",
                "passes_quiet": False,
            }
        ]
    )
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    timer_watch.fire_due(store, dif, now=NOW)
    assert dif.device.call_args.kwargs["release_pending"] is False
