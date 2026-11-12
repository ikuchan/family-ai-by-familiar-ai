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


def _tool(*, speaker: str = "パパ"):
    store = _FakeStore()
    oif = MagicMock()
    oif.write = AsyncMock(return_value="obs-1")
    hushed: list = []
    unhushed: list = []

    async def hush(person, until, tid):
        hushed.append((person, until, tid))

    async def unhush(tid):
        unhushed.append(tid)

    t = TimerTool(
        store=lambda: store,
        oif=oif,
        speaker=lambda: speaker,
        quiet=lambda: QuietHoursRule(23, 7),
        silence_active=lambda: False,
        now=lambda: NOW,
        hush=hush,
        unhush=unhush,
    )
    return t, hushed, unhushed


def test_setting_a_timer_hushes_until_due():
    t, hushed, _ = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and "黙って" in text
    assert hushed == [("パパ", NOW + timedelta(minutes=3), 1)]


def test_a_timer_hushes_even_when_no_one_is_known_to_have_asked():
    """知-ai（2026-10-05）：頼んだ人が分からなくても黙る。以前は話者が分かるときだけ黙り、耳を閉じる仕組み
    （`TIMER_MIC_CLOSE`）は人に関係なく掛かるので、耳は閉じて口は開いたままになった（実機 09-26 08:52）。"""
    t, hushed, _ = _tool(speaker="")
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and "黙って" in text
    assert hushed == [("", NOW + timedelta(minutes=3), 1)]


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


# ── 環-aw 第 2 段 項目 4：タイマーの沈黙も状態図へ送る（持ち主は状態図）──────────────────────────


def _agent_with_chart():
    from types import SimpleNamespace

    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop.paju_state import PajuState

    sm = PajuState()
    a = SimpleNamespace(paju_state=sm)
    hush = lambda person, due, tid: EmbodiedAgent._hush_for_timer(a, person, due, tid)  # noqa: E731
    unhush = lambda tid: EmbodiedAgent._unhush_timer(a, tid)  # noqa: E731
    return sm, hush, unhush


def _due(minutes: float) -> datetime:
    return datetime.now(ZoneInfo("Asia/Tokyo")) + timedelta(minutes=minutes)


def test_a_timer_through_the_agent_silences_the_chart():
    async def go():
        sm, hush, unhush = _agent_with_chart()
        await hush("パパ", _due(3), 1)
        assert sm.is_silent() and sm.silence_request().reason == "timer:1"
        await unhush(2)  # 違うタイマーでは解かない
        assert sm.is_silent()
        await unhush(1)
        assert not sm.is_silent() and sm.silence_request() is None

    asyncio.run(go())


def test_a_timer_through_the_agent_does_not_overwrite_a_spoken_request():
    async def go():
        sm, hush, unhush = _agent_with_chart()
        spoken = SilenceRequest(person="パパ", until=time.time() + 600)
        await sm.hush_with(spoken)
        await hush("パパ", _due(3), 1)
        assert sm.silence_request() == spoken
        await unhush("all")  # 人の依頼は、タイマーを止めても解けない
        assert sm.silence_request() == spoken

    asyncio.run(go())


def test_a_later_timer_through_the_agent_extends_the_silence():
    async def go():
        sm, hush, _unhush = _agent_with_chart()
        await hush("パパ", _due(3), 1)
        await hush("パパ", _due(8), 2)
        assert sm.silence_request().until == pytest.approx(_due(8).timestamp(), abs=2)
        assert (
            sm.silence_request().reason == "timer:2"
        )  # 遅いタイマーの依頼に置き換わる（純関数の決まり）

    asyncio.run(go())


def test_without_a_chart_the_timer_silence_does_nothing():
    from types import SimpleNamespace

    from familiar_agent.agent import EmbodiedAgent

    a = SimpleNamespace(paju_state=None)
    asyncio.run(EmbodiedAgent._hush_for_timer(a, "パパ", _due(3), 1))
    asyncio.run(EmbodiedAgent._unhush_timer(a, 1))
    assert EmbodiedAgent._silence_active_now(a) is False


def test_a_timer_before_the_chart_has_started_keeps_a_restored_spoken_request():
    """再起動のすぐあと、状態図を始める前にタイマーを掛けても、控えの人の依頼を戻してから決める（上書きしない）。"""
    from types import SimpleNamespace

    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop.paju_state import PajuState

    spoken = SilenceRequest(person="パパ", until=time.time() + 3600)

    async def go():
        sm = PajuState(restore_silence=lambda: spoken)
        a = SimpleNamespace(paju_state=sm)
        await EmbodiedAgent._hush_for_timer(a, "パパ", _due(3), 1)
        assert sm.silence_request() == spoken

    asyncio.run(go())


def test_the_chart_exists_as_soon_as_the_loop_is_made():
    """`/timer stop` は門より前に agent が直に処理する。ループを作った時点で状態図があり、`agent.paju_state` に置かれる。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    assert a.paju_state is ip._paju()


def test_a_timer_stopped_before_any_input_clears_its_restored_silence(monkeypatch):
    """再起動のすぐあと、最初の入力の前に `/timer stop` で止めても、控えのタイマーの沈黙を戻してから解き、控えも消す。"""
    import familiar_agent.silence_state as ss
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ss.save_silence(SilenceRequest(person="パパ", until=time.time() + 600, reason="timer:1"))
    a = _agent(stream_returns=[])
    InformationProcessing(a)  # 起動したループ（まだ何も届いていない）
    asyncio.run(EmbodiedAgent._unhush_timer(a, 1))
    assert ss.load_silence() is None
    assert a.paju_state.silence_request() is None
