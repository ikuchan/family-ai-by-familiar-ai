"""つなぎの声が鳴り終わってから 1 秒は、答えを声にしない（出-aq 段 5・2026-09-29）。

つなぎは答えまで時間がかかったときだけ出す。出した直後に答えが届くと、「ちょっと待ってね」の
すぐ後に答えが始まり、つなぎの意味が無くなる。本人「つなぎを言ったらそこから数秒発話を待たせる」。
数え始めは**声が鳴り終わったとき**（声は背景で鳴り、長さは文で違う）、待つのは 1 秒（本人の決定）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

GAP = 0.3  # 試験では短くする（既定は 1.0 秒・`filler_answer_gap_seconds`）
FILLER_VOICE = 0.2  # つなぎの声が鳴る長さ（偽物）


def _ip(spoken: list):
    a = _agent(stream_returns=[])
    a.config.filler_answer_gap_seconds = GAP
    ip = InformationProcessing(a)
    ip._wake_window().open(time.monotonic())  # 入口を通った会話として窓を開けておく（出-as 段 4）
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]

    async def speak(text, **_kw):
        spoken.append(("始", text, time.monotonic()))
        try:
            if "待って" in text:
                await asyncio.sleep(FILLER_VOICE)
        finally:  # 止められても、鳴り終わった時刻は残す
            spoken.append(("終", text, time.monotonic()))

    ip._dif = MagicMock(speak=speak)
    return ip


def _at(spoken, edge, text):
    return next((t for e, s, t in spoken if e == edge and s == text), None)


def test_an_answer_ready_at_once_waits_until_a_second_after_the_filler_voice() -> None:
    spoken: list = []
    ip = _ip(spoken)

    async def scenario():
        await ip._say_filler("ちょっと待ってね。")
        await ip._speak("明日は晴れです。")  # 答えはもう届いている
        await ip.close()

    asyncio.run(scenario())
    filler_end = _at(spoken, "終", "ちょっと待ってね。")
    answer_start = _at(spoken, "始", "明日は晴れです。")
    assert filler_end is not None, f"つなぎが鳴る前に答えが出た：{spoken}"
    assert answer_start - filler_end >= GAP - 0.02


def test_an_answer_after_the_gap_does_not_wait() -> None:
    spoken: list = []
    ip = _ip(spoken)

    async def scenario():
        await ip._say_filler("ちょっと待ってね。")
        await asyncio.sleep(FILLER_VOICE + GAP + 0.1)  # 答えが届いたのは、間が過ぎてから
        asked = time.monotonic()
        await ip._speak("明日は晴れです。")
        await ip.close()
        return asked

    asked = asyncio.run(scenario())
    assert _at(spoken, "始", "明日は晴れです。") - asked < 0.05


def test_without_a_filler_the_answer_does_not_wait() -> None:
    spoken: list = []
    ip = _ip(spoken)

    async def scenario():
        asked = time.monotonic()
        await ip._speak("明日は晴れです。")
        await ip.close()
        return asked

    asked = asyncio.run(scenario())
    assert _at(spoken, "始", "明日は晴れです。") - asked < 0.05
