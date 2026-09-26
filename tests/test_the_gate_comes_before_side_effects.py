"""声は門を通ってから、初めて何かを起こす（出-au 段 1-2・1-3・2026-09-26・`設計方針_判定の段` §2.1・§2.4）。

以前は `push_utterance` が入口の門より前に打ち切り（`_abort_lookups`）と話しかけられた時刻の印を付けて
いたので、窓の外で捨てる声（テレビ・家族の話）でも、調べもの・考えている途中の主LLM・情動の求め・
タイマーの知らせの求めが取り消され、「ひとりの回数」が 0 に戻った。

- 門は `push_utterance` の中で、**届いた時刻**で判定する（駆動体が取り出した時刻ではない）。
- 窓は **30 秒**。**声もキーボードも**名前で開く。窓の外の入力は捨て、副作用を起こさない。
- **打ち切りは名前がある入力でだけ。** タイマーの操作も名前が要る（「パジュ、止めて」）。
- 名前の無い入力は、飛行中の調べもの（主LLM を含む）がある間は待たせ、前の求めの答えが出てから別の求めに
  する（段 3 で「前の求めに添える」を入れるまでの既定・Jev が使えないときの「そのまま出す」と同じ）。
"""

from __future__ import annotations

import pytest

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.silence_hold import lifts
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.request import Lookup

from tests.test_event_loop import _agent

pytestmark = pytest.mark.real_window  # 門そのものを確かめる（conftest の窓の開き口を使わない）


def _ip(*, timer: bool = False):
    a = _agent(stream_returns=[])
    a.config.agent_names = ["パジュ"]
    a._social_presence_permission = MagicMock(return_value=1.0)
    a._nudge_seeking = AsyncMock()
    a._last_human_at = None
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None
    ip._ensure_driver = lambda: None  # 駆動体を起こさず、積んだものだけを見る
    ip._abort_lookups = AsyncMock()
    ip._timer_active = lambda: timer
    return ip, a


def _push(ip, text, *, source="voice", arrived=None):
    """積まれたら True（返事を待っている）、捨てられたら False。"""

    async def go():
        task = asyncio.ensure_future(ip.push_utterance(text, source=source, arrived=arrived))
        await asyncio.sleep(0.01)
        queued = not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return queued

    return asyncio.run(go())


# ── 窓の外の入力は、何も起こさない ─────────────────────────────────────────


def test_a_voice_outside_the_window_changes_nothing():
    ip, a = _ip()
    assert _push(ip, "ごちそうさまでした") is False
    ip._abort_lookups.assert_not_awaited()  # 調べものも主LLM も止めない
    assert a._last_human_at is None  # ひとりの回数を戻さない
    assert ip._triggers.empty()


def test_the_keyboard_also_needs_the_name():
    ip, a = _ip()
    assert _push(ip, "明日の天気は？", source="keyboard") is False
    assert _push(ip, "パジュ、明日の天気は？", source="keyboard") is True


# ── 名前がある入力だけが打ち切る ─────────────────────────────────────────


def test_a_named_input_is_heard_and_interrupts():
    ip, a = _ip()
    assert _push(ip, "パジュ、いや明後日の天気") is True
    ip._abort_lookups.assert_awaited_once()
    assert a._last_human_at is not None


def test_an_unnamed_input_inside_the_window_is_heard_without_interrupting():
    ip, a = _ip()
    ip._wake_window().open(time.monotonic())
    assert _push(ip, "うん") is True
    ip._abort_lookups.assert_not_awaited()
    assert a._last_human_at is not None


# ── 窓は 30 秒・届いた時刻で ─────────────────────────────────────────────


def test_the_window_is_thirty_seconds():
    ip, _ = _ip()
    ip._wake_window().open(100.0)
    assert _push(ip, "うん", arrived=129.0) is True
    ip2, _ = _ip()
    ip2._wake_window().open(100.0)
    assert _push(ip2, "うん", arrived=131.0) is False


def test_the_window_is_judged_by_when_it_arrived_not_when_it_is_taken():
    """画面が前の `run()` を待っていて遅れて渡しても、届いた時刻が窓の中なら受ける。"""
    ip, _ = _ip()
    ip._wake_window().open(100.0)  # 窓は 130 まで。いまの時計はそれよりずっと後
    assert _push(ip, "うん", arrived=120.0) is True


# ── タイマーの操作も名前が要る ───────────────────────────────────────────


def test_a_timer_control_word_needs_the_name():
    ip, _ = _ip(timer=True)
    assert _push(ip, "止めて") is False
    assert _push(ip, "パジュ、止めて") is True
    ip._abort_lookups.assert_awaited_once()


def test_a_timer_silence_is_lifted_only_by_the_named_control_word():
    names = ["パジュ"]
    assert lifts("会話入力", "止めて", names=names, reason="timer:1") is False
    assert lifts("会話入力", "パジュ、止めて", names=names, reason="timer:1") is True


# ── 名前の無い入力は、飛行中の調べものがある間は待つ ─────────────────────


def _convo(text, *, named):
    return Trigger(kind="会話入力", query=text, source="voice", named=named)


def test_an_unnamed_input_waits_while_something_is_in_flight():
    ip, _ = _ip()
    ip._req.request_id = "req-1"
    ip._req.lookups.append(Lookup(index=1, action="主LLM", query="主LLM1", generation=0))
    ip._triggers.put_nowait(_convo("うん", named=False))
    ip._triggers.put_nowait(Trigger(kind="完了", query="主LLM1", result="{}"))
    got = asyncio.run(ip._take_trigger())
    assert got is None  # 先に前の求めの完了を取り込む
    assert [t.query for t in ip._held] == ["うん"]


def test_the_waiting_input_goes_first_once_nothing_is_in_flight():
    ip, _ = _ip()
    ip._held.append(_convo("うん", named=False))
    ip._held.append(Trigger(kind="情動", query="bond"))
    got = asyncio.run(ip._take_trigger())
    assert got is not None and got.query == "うん"


def test_a_named_input_does_not_wait():
    ip, _ = _ip()
    ip._req.request_id = "req-1"
    ip._req.lookups.append(Lookup(index=1, action="主LLM", query="主LLM1", generation=0))
    ip._triggers.put_nowait(_convo("パジュ、いや明後日", named=True))
    got = asyncio.run(ip._take_trigger())
    assert got is not None and got.query == "パジュ、いや明後日"
