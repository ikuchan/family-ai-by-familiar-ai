"""名前で受けた話しかけは、返事の話し終わりから窓を開ける（出-cx・2026-10-21・本人の決定）。

窓は話し始めから 10 秒なので、名前つきで 10 秒を超えて話すと、届いた時点で窓はもう閉じている。返事を話しても
保持（`hold`・`release`）は開いているときしか効かず、返事のあとの名前なしの「ありがとう」「それと…」を窓の外として
捨てていた。考えている最中には開けない（返事の前に別の話を受けない・本人「マルチスレッドな対応が困る」）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tests.test_the_window_governs_the_reply import _Clock, _ip


def _closed_ip(monkeypatch, *, named: bool, kind: str = "発話", speech_sec: float = 6.0):
    """窓は閉じている（45 秒前に開いて切れた）。確認待ちではない。声は `speech_sec` 秒かかる。"""
    a, ip = _ip(kind, window_open=False, monkeypatch=monkeypatch)
    a.confirm_alive = lambda: False
    ip._req.named = named
    clock = _Clock(1000.0)
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.monotonic", clock)

    async def speak(text, **kw):
        clock.t += speech_sec

    ip._dif.speak = AsyncMock(side_effect=speak)
    return ip


def test_a_named_request_opens_the_window_after_the_reply(monkeypatch):
    ip = _closed_ip(monkeypatch, named=True)
    spoken, outcome = asyncio.run(ip._speak("晴れだよ"))
    assert outcome == "発話"
    assert ip._wake.is_open(1006.0 + 9.0)  # 話し終わり（1006）から 9 秒後も開いている


def test_the_reopened_window_still_closes_ten_seconds_after(monkeypatch):
    """延ばしすぎない：話し終わりから 10 秒で閉じる。"""
    ip = _closed_ip(monkeypatch, named=True)
    asyncio.run(ip._speak("晴れだよ"))
    assert not ip._wake.is_open(1006.0 + 10.0)


def test_an_unnamed_request_leaves_a_closed_window_closed(monkeypatch):
    """**反証の側**：名前なしで受けた求め（窓の中で受け、返事の前に閉じた）は、今までどおり開け直さない。"""
    ip = _closed_ip(monkeypatch, named=False)
    asyncio.run(ip._speak("晴れだよ"))
    assert not ip._wake.is_open(1006.0)


def test_affect_and_device_speech_leave_a_closed_window_closed(monkeypatch):
    """**反証の側**：会話の求めでなければ、名前の印が残っていても開け直さない。"""
    for kind in ("情動", "機器"):
        ip = _closed_ip(monkeypatch, named=True, kind=kind)
        ip._req.request_text = "[入室] パパ が来た"
        asyncio.run(ip._speak("おかえりなさい"))
        assert not ip._wake.is_open(1006.0), kind


def _begin(ip, kind: str) -> None:
    ip._write_origin = AsyncMock(return_value="obs-1")  # type: ignore[method-assign]
    ip._note_origin = MagicMock()  # type: ignore[method-assign]
    ip._notify_request_state = MagicMock()  # type: ignore[method-assign]
    asyncio.run(ip._begin_request(kind=kind, text="パジュ、晴れる？", utterance="パジュ、晴れる？"))


def test_the_request_keeps_whether_it_was_named(monkeypatch):
    """名前で呼ばれたか（`Trigger.named`）は、`_begin_utterance` が置き、`_begin_request` が求めへ写す。"""
    _, ip = _ip(monkeypatch=monkeypatch)
    ip._pending_named = True  # `_begin_utterance` が置く
    _begin(ip, "発話")
    assert ip._req.named is True
    _begin(ip, "発話")  # 写したら空に戻す（次の求めへ持ち越さない）
    assert ip._req.named is False
    ip._pending_named = True
    _begin(ip, "情動")  # 会話でなければ写さない
    assert ip._req.named is False


def test_the_utterance_start_hands_over_the_name():
    """`_begin_utterance` が、きっかけの名前の印を `_begin_request` へ渡す（`_pending_source` と同じ口）。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._begin_utterance)
    assert "self._pending_named = trigger.named" in src
