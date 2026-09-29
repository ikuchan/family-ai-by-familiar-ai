"""同じつなぎを 2 度言わない（出-aq 段 4・2026-09-29）。

つなぎは待たせているあいだ 20 秒ごとに出る（出-au 段 2）。軽量LLM には「すでに相手へ伝えた一言」を
渡しているが、同じことを 2 回言わせない指示は通らなかった（`根拠台帳` §47）。本人「同じ言葉を何回も
言うのは機械的すぎて不可」。**言う直前に機械が照らす。** 記号と空白を外し、どちらかがもう片方を
まるごと含んでいたら同じとみなす（本人の決定ア）。言い換えは落とさない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.filler_echo import repeats
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_the_same_words_with_other_marks_repeat() -> None:
    assert repeats("ちょっと待ってね！", ["ちょっと待ってね。"])


def test_words_that_hold_an_earlier_filler_repeat() -> None:
    assert repeats("いま調べています", ["調べています。"])
    assert repeats("調べています", ["いま調べています。"])


def test_a_rewording_is_not_a_repeat() -> None:
    assert not repeats("もう少し待ってね", ["ちょっと待ってね。"])


def test_nothing_said_yet_is_not_a_repeat() -> None:
    assert not repeats("ちょっと待ってね", [])
    assert not repeats("。", ["ちょっと待ってね"])  # 記号だけの文は照らさない


def _ip():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._wake_window().open(time.monotonic())  # 入口を通った会話として窓を開けておく（出-as 段 4）
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._dif = MagicMock(speak=AsyncMock())
    return ip


def test_a_repeated_filler_is_neither_spoken_nor_kept() -> None:
    ip = _ip()
    ip._agent._oif.write = AsyncMock(return_value="obs")  # O へ何回書いたかを数える

    async def scenario():
        await ip._say_filler("ちょっと待ってね。")
        await ip._say_filler("ちょっと待ってね！")
        await asyncio.sleep(0)  # 背景の声を走らせる（出-aq 段 1）
        n = ip._dif.speak.await_count
        said = list(ip._req.said_fillers)
        writes = ip._agent._oif.write.await_count
        await ip.close()
        return n, said, writes

    n, said, writes = asyncio.run(scenario())
    assert n == 1
    assert said == ["ちょっと待ってね。"]
    assert writes == 1  # O にも 1 回だけ


def test_a_different_filler_still_goes_out() -> None:
    ip = _ip()

    async def scenario():
        await ip._say_filler("ちょっと待ってね。")
        await ip._say_filler("もう少しです。")
        await asyncio.sleep(0)
        n = ip._dif.speak.await_count
        await ip.close()
        return n

    assert asyncio.run(scenario()) == 2
