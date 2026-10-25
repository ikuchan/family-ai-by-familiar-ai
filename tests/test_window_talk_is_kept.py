"""ループは、窓の会話の行を出来事が起きたその場で足す（環-ar 段 A・2026-10-25・本人の決定）。

足すのは、人の言葉を受けた（添えた言葉も）・返事を言った・調べた結果が届いた、の 3 つ。つなぎは O に書いていないので
足さない。窓が閉じてから開き直すと、新しい会話になる。返事の話し終わりに開け直すとき（出-cx・出-cw）は同じ会話の続き。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.intent_ledger import IntentLedger
from tests.test_event_loop import _agent
from tests.test_intent_ledger import _Store


def _ip():
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._current_speaker_name = lambda: "パパ"
    ip._wake_window().open(time.monotonic())
    return ip


def _lines(ip):
    return [line.split(" ", 2)[2] for line in ip._window_talk().lines]


def test_an_accepted_utterance_is_a_line():
    ip = _ip()
    ip._begin_request = AsyncMock()
    ip._iterate = AsyncMock(return_value="")
    asyncio.run(ip._utterance_iteration("しりとりしよう"))
    assert _lines(ip) == ["パパ：しりとりしよう"]


def test_added_words_are_a_line():
    ip = _ip()
    ip._write_origin = AsyncMock(return_value="obs-1")
    asyncio.run(ip._attach_to_request(Trigger(kind="会話入力", query="東京の")))
    assert _lines(ip) == ["パパ：東京の"]


def test_a_spoken_answer_is_a_line_and_an_unspoken_one_is_not():
    ip = _ip()
    ip._req.trigger_kind = "発話"

    async def go():
        await ip._finish("めだか！", [], "発話")
        await ip._finish("（考えただけ）", [], "沈黙")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    assert _lines(ip) == ["わたし（パパへ）：めだか！"]


def test_a_returned_result_is_one_short_line():
    ip = _ip()
    ip._ledger = IntentLedger(store=_Store())
    ip._req.request_id, ip._req.trigger_kind = "req-1", "発話"
    lk = ip._open_lookup("search_deferred", "明日の天気")
    ip._drained_completions.append(
        Trigger(kind="完了", query="明日の天気", result="晴れ。" * 60, intent_id=lk.intent_id)
    )
    ip._write_version = AsyncMock(return_value="ver-1")
    asyncio.run(ip._intake())
    (line,) = _lines(ip)
    assert line.startswith("わたし：調べた（明日の天気）：晴れ。")
    assert len(line) < 120


def test_a_new_window_starts_an_empty_talk():
    """**反証の側**：窓が閉じてから開き直すと、前の会話の行は載らない。"""
    ip = _ip()
    ip._talk_line("パパ", "りす")
    first = ip._window_talk()
    ip._wake_window().close()
    ip._wake_window().open(time.monotonic())
    assert ip._window_talk() is not first and ip._window_talk().lines == []


def test_the_text_is_given_only_while_the_window_is_open():
    ip = _ip()
    ip._talk_line("パパ", "りす")
    assert "パパ：りす" in ip._window_text()
    ip._wake_window().close()
    assert ip._window_text() == ""


# --- 見直しの指摘（窓の外のものを、閉じた会話に足さない）---


def test_a_confirm_long_after_does_not_revive_an_old_talk():
    """窓の外で始まった求めの確認の問いが窓を開けたら、新しい会話にする（古い会話の続きにしない）。"""
    from unittest.mock import MagicMock

    ip = _ip()
    ip._talk_line("パパ", "りす")
    old = ip._wake_window().session
    ip._wake_window().close()
    ip._req.window_session = 0  # 窓の外で始まった求め（タイマーなど）
    ip._dif = MagicMock(speak=AsyncMock())
    ip._confirming = lambda: True
    asyncio.run(ip._say_aloud("3 分のタイマーね。いい？"))
    assert ip._wake_window().session == old + 1


def test_a_reply_to_a_request_begun_in_the_window_continues_its_talk():
    """**反証の側**：窓の中で始まった名前つきの頼みの返事は、窓が切れていても同じ会話の続き（出-cx）。"""
    from unittest.mock import MagicMock

    ip = _ip()
    session = ip._wake_window().session
    ip._req.window_session, ip._req.trigger_kind, ip._req.named = session, "発話", True
    ip._wake_window().close()
    ip._dif = MagicMock(speak=AsyncMock())
    ip._confirming = lambda: False
    asyncio.run(ip._say_aloud("いいよ。しりとり！"))
    assert ip._wake_window().session == session


def test_lines_from_outside_the_window_are_not_added():
    ip = _ip()
    ip._wake_window().close()
    ip._req.window_session = 0
    ip._talk_line("わたし", "遅れて届いた")
    assert ip._window_talk().lines == []


def test_a_failed_lookup_and_a_timer_voice_are_not_lines():
    ip = _ip()
    ip._ledger = IntentLedger(store=_Store())
    ip._req.request_id, ip._req.trigger_kind = "req-1", "発話"
    lk = ip._open_lookup("search_deferred", "明日の天気")
    ip._drained_completions.append(
        Trigger(kind="完了", query="明日の天気", result="x", intent_id=lk.intent_id, failed=True)
    )
    ip._write_version = AsyncMock(return_value="ver-1")
    asyncio.run(ip._intake())
    ip._req.trigger_kind = "機器"
    ip._talking = lambda: False

    async def go():
        await ip._finish("3 分たったよ", [], "発話")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    assert ip._window_talk().lines == []
