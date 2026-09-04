"""調べものが遅いとき、**1回だけ**「まだかかっている」を知らせる（案G-3・案イ・案ハ）。

`search_deferred` は実測で平均2.5秒・最長22.1秒。つなぎを一言だけ言って22秒黙るのは
落ち着かない。時計で定期的に起こすのではなく、**遅いという事実**を1回きりの起点にする。

「進捗」は結果ではないので、飛行中の数も一覧も触らず、意図も supersede しない。受けた
反復は**つなぎだけ出して閉じない**（閉じると、あとから届く結果に行き場が無くなる）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _turn


def test_the_threshold_comes_from_config():
    import os
    from unittest.mock import patch

    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().lookup_slow_seconds == 5.0


def test_a_slow_lookup_raises_a_progress_event_once():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a.config.lookup_slow_seconds = 0.01

    async def scenario():
        ip = InformationProcessing(a)
        ip._in_flight_lookups = [("search_deferred", "明日の天気", 1)]
        await ip._watch_slow_lookup("明日の天気", ip._generation)
        return ip

    ip = asyncio.run(scenario())
    assert ip._completion_queue.qsize() == 1
    assert ip._completion_queue.get_nowait()[3] == "進捗"


def test_no_progress_event_once_the_result_has_arrived():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a.config.lookup_slow_seconds = 0.01

    async def scenario():
        ip = InformationProcessing(a)
        ip._in_flight_lookups = []            # もう結果が来ている
        await ip._watch_slow_lookup("明日の天気", ip._generation)
        return ip

    assert asyncio.run(scenario())._completion_queue.empty()


def test_no_progress_event_for_an_abandoned_request():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a.config.lookup_slow_seconds = 0.01

    async def scenario():
        ip = InformationProcessing(a)
        ip._in_flight_lookups = [("search_deferred", "明日の天気", 1)]
        ip._generation = 1                    # 見張りを立てたあとに打ち切られた
        await ip._watch_slow_lookup("明日の天気", 0)
        return ip

    assert asyncio.run(scenario())._completion_queue.empty()


def test_a_progress_iteration_only_says_a_filler():
    # つなぎだけ出して閉じない。飛行中の数も触らない。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "本応答"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"full","effort":"high","text":"もう少しかかりそうです"}')
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        ip.set_output(shown.append)
        ip._utterance = "明日の天気は？"
        ip._inflight = 1
        ip._in_flight_lookups = [("search_deferred", "明日の天気", 1)]
        ip._completion_queue.put_nowait(("明日の天気", "", None, "進捗", 0))
        await ip._iterate()
        await ip.close()
        return ip

    ip = asyncio.run(scenario())
    assert "もう少しかかりそうです" in "".join(shown)
    assert "本応答" not in "".join(shown)          # 閉じない
    assert ip._inflight == 1                      # 飛行中のまま
    assert ip._in_flight_lookups == [("search_deferred", "明日の天気", 1)]
