"""確認待ちのあいだに話したら、話し終わりから窓を開ける（出-cw・2026-10-18・本人の決定イ）。

10/18 07:55 に、名前つきの頼みがまわりの声と続いて 12.9 秒の音になり、窓は話し始め（07:55:20 ごろ）から 10 秒で
閉じていた。聞き返し（「はいか、いいえ…」）を話し始めた 07:55:34 には閉じていたので延びず、07:55:45 の「はい」を
窓の外として捨てた。確認の問いと聞き返しには、名前なしで答えられるようにする。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.test_the_window_governs_the_reply import _Clock, _ip


def _confirming_ip(monkeypatch, *, confirming: bool = True, speech_sec: float = 6.0):
    """窓は閉じている（45 秒前に開いて切れた）。声は `speech_sec` 秒かかる。"""
    a, ip = _ip(window_open=False, monkeypatch=monkeypatch)
    a.confirm_alive = lambda: confirming
    clock = _Clock(1000.0)
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.monotonic", clock)

    async def speak(text, **kw):
        clock.t += speech_sec

    ip._dif.speak = AsyncMock(side_effect=speak)
    return ip


def test_the_confirm_question_opens_the_window_after_it_ends(monkeypatch):
    ip = _confirming_ip(monkeypatch)
    pc = SimpleNamespace(text="15 分のタイマーね。その間は黙って聞かないよ、いい？")
    asyncio.run(ip._ask_confirm_aloud(pc))
    ip._dif.speak.assert_awaited_once()
    assert ip._wake.is_open(1006.0 + 9.0)  # 話し終わり（1006）から 9 秒後も開いている


def test_the_ask_back_opens_the_window_after_it_ends(monkeypatch):
    ip = _confirming_ip(monkeypatch)
    spoken, outcome = asyncio.run(
        ip._speak("いまタイマーを掛けるか確かめているところだよ。はいか、いいえで")
    )
    assert outcome == "発話"
    assert ip._wake.is_open(1006.0 + 9.0)


def test_without_a_pending_confirm_a_closed_window_stays_closed(monkeypatch):
    """**反証の側**：確認待ちでなければ、閉じた窓は声で開け直さない（今までの決まり）。"""
    ip = _confirming_ip(monkeypatch, confirming=False)
    asyncio.run(ip._speak("晴れだよ"))
    assert not ip._wake.is_open(1006.0)


def test_the_reopened_window_still_closes_ten_seconds_after(monkeypatch):
    """延ばしすぎない：話し終わりから 10 秒で閉じる。"""
    ip = _confirming_ip(monkeypatch)
    asyncio.run(ip._speak("はいか、いいえで"))
    assert not ip._wake.is_open(1006.0 + 10.0)
