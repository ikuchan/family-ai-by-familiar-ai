"""声はどれも 1 つの口から、普段の声の倍率（`TTS_GAIN`）を付けて出す（2026-10-07 実機 23:18:39）。

つなぎの声（出-aq 段 1・09-25）は、倍率（環-q-ろ・09-19・既定 0.25）より後に入り、`dif.speak(text)` と倍率を渡さずに
鳴らしていた。「承知いたしました。少々お待ちください。」だけ普段の 4 倍の声で鳴った（本人）。声を出す口を
`_say_aloud` の 1 か所にまとめ、倍率はそこで必ず付ける。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import event_loop
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    a.config.tts_gain = 0.25
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    ip._req.request_text = "パジュ、音楽をかけて"
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()
    return ip


def test_the_filler_voice_gets_the_everyday_gain():
    ip = _ip()

    async def run():
        ip._speak_filler_in_background("承知いたしました。少々お待ちください。")
        await asyncio.wait(list(ip._filler_voices))

    asyncio.run(run())
    ip._dif.speak.assert_awaited_once()
    assert ip._dif.speak.await_args.kwargs.get("gain") == 0.25


def test_the_reply_and_the_filler_get_the_same_gain():
    ip = _ip()
    ip._delivery_block_reason = lambda: ""
    ip._wake.open(event_loop.time.monotonic())

    async def run():
        ip._speak_filler_in_background("少々お待ちください。")
        await asyncio.wait(list(ip._filler_voices))
        await ip._speak("はい、何の曲にしますか？")

    asyncio.run(run())
    gains = [c.kwargs.get("gain") for c in ip._dif.speak.await_args_list]
    assert gains == [0.25, 0.25]


def test_only_one_place_calls_the_speaker():
    """声を出す口は 1 か所（`_say_aloud`）。直接 `dif.speak` を呼ぶ道を増やすと、また倍率を渡し忘れる。"""
    src = inspect.getsource(event_loop)
    assert src.count("self._dif.speak(") == 1
    assert "self._dif.speak(" in inspect.getsource(InformationProcessing._say_aloud)
