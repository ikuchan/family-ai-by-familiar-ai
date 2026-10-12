"""つなぎは書き手と声を裏の仕事に出し、反復はすぐ閉じる（出-bx・2026-10-13・本人の決定）。

10/12 23:59 の天気の質問で、見張りがつなぎを頼んだ 0.04 秒あとに検索の結果が届いた。駆動体は 1 反復ずつ回すので、
つなぎの反復（想起・書き手 1.05 秒）が終わるまで結果の取込が始まらず、1.33 秒待たされた。つなぎの反復は想起だけして、
書き手と声を裏の仕事に出す。答えの声を出し始めたら、まだ声になっていないつなぎは言わない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, write_takes: float = 0.2):
    ip = InformationProcessing(_agent(stream_returns=[]))
    said: list[str] = []
    ip._waiting_on = lambda: [MagicMock(action="主LLM")]  # type: ignore[method-assign]

    async def write(inp, what, avoid=()):
        await asyncio.sleep(write_takes)
        return "もう少しね"

    async def say(text):
        said.append(text)

    ip._say_filler = say  # type: ignore[method-assign]
    ip._arbiter = lambda: MagicMock(write_filler=write)  # type: ignore[method-assign]
    ip._arbiter_input = lambda **k: None  # type: ignore[method-assign]
    ip._req.filler_pending = True
    return ip, said


def test_the_iteration_does_not_wait_for_the_writer():
    ip, said = _ip()
    took: list[float] = []

    async def go():
        started = time.monotonic()
        await ip._say_waiting_filler("x", "", 1)
        took.append(time.monotonic() - started)
        assert said == []  # まだ書いている
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert took[0] < 0.1
    assert said == ["もう少しね"]
    assert ip._req.filler_pending is False


def test_a_filler_finished_after_the_answer_began_is_not_said(caplog):
    ip, said = _ip()

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        ip._req.answering = True  # 書いているあいだに答えの声が始まった
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert said == []
    assert "答えを言い始めたのでつなぎを言わない" in caplog.text
    assert ip._req.filler_pending is False


def test_a_filler_of_a_cut_request_is_not_said():
    """**反証の側**：書いているあいだに求めが打ち切られたら言わない。"""
    ip, said = _ip()

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        ip._request_generation += 1
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert said == []


def test_the_answer_marks_that_it_began():
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]

    async def go():
        await ip._speak("明日は晴れです")
        await ip.close()

    asyncio.run(go())
    assert ip._req.answering is True


def test_close_cancels_a_filler_still_being_written():
    ip, said = _ip(write_takes=5.0)
    jobs: list = []

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        jobs.extend(ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert said == [] and all(j.done() for j in jobs)
