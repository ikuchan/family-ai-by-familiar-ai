"""停止ボタンは駆動体の求めへ届く（環-j・2026-09-13）。

会話入力の `Future` は最初の反復で解決するので、調べものを投げた時点で GUI のターンは終わり、
停止ボタンは無効になっていた。続きの反復（完了→調停→主LLM→発話）は駆動体の上で回り、
待ち手が居ない。ループが「求めが開いているか」を通知し、GUI はそのあいだ停止を有効にして、
押されたら `abort_current()` で求めを丸ごと打ち切る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    return a, ip


def test_the_loop_tells_when_a_request_opens_and_closes() -> None:
    states: list[bool] = []

    async def scenario():
        a, ip = _ip()
        ip.set_request_state_listener(states.append)
        await ip._begin_request(kind="発話", text="やあ", utterance="やあ")
        opened = list(states)
        await ip._finish("", [], "沈黙")
        await ip.close()
        return opened, list(states)

    opened, final = asyncio.run(scenario())
    assert opened == [True]
    assert final == [True, False]


def test_abort_current_cancels_the_open_request_and_tells() -> None:
    states: list[bool] = []

    async def scenario():
        a, ip = _ip()
        ip.set_request_state_listener(states.append)
        await ip._begin_request(kind="発話", text="明日の天気は？", utterance="明日の天気は？")
        gen = ip._request_generation
        ip._req.lookups.append(
            MagicMock(in_flight=True, index=1, action="search_deferred", query="天気")
        )
        never = asyncio.create_task(asyncio.sleep(3600))
        ip._background_tasks.add(never)
        await ip.abort_current(reason="停止ボタン")
        await asyncio.sleep(0)
        got = (ip._request_generation > gen, never.cancelled(), ip._req.request_id, list(states))
        await ip.close()
        return got

    bumped, cancelled, request_id, final = asyncio.run(scenario())
    assert bumped and cancelled and request_id is None
    assert final == [True, False]


def test_abort_current_with_nothing_open_is_harmless() -> None:
    async def scenario():
        a, ip = _ip()
        await ip.abort_current(reason="停止ボタン")
        await ip.close()

    asyncio.run(scenario())


def test_the_agent_exposes_interrupt() -> None:
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._info_processing = MagicMock()
    a._info_processing.abort_current = AsyncMock()
    asyncio.run(EmbodiedAgent.interrupt(a, reason="停止ボタン"))
    a._info_processing.abort_current.assert_awaited_once_with(reason="停止ボタン")
