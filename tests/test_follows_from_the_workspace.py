"""そのターンが何に続くかを、W の中から選ばせる（段 4・判定は軽量LLM）。

**続き先は、その記録を作るのに使ったものの中にしかない。** W に並んでいるのは、そのとき
頭にあったものである。そこから一つ選ぶのが、生成のパターンを残すということである
（`設計方針_MI間の関係`）。

判定は軽量LLM の専用の仕事である（`根拠台帳` §29）。主LLM に `say` の欄で名指させる形は
続きの場面の 36.1% しか返さず、言い方を強めると偽陽性が増えた。ここでは、判定が返した
id をどう扱うかを見る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _run, _turn


def _judge(a, verdict):
    a._evaluator.judge_follows = AsyncMock(return_value=verdict)


def test_the_named_memory_becomes_what_this_turn_follows():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _judge(a, "m1")
    _run(a, utterance="さっきの話だけど")
    # 前＝判定が返した記憶、後＝このターンの起点。
    assert a._memory.record_succession.call_args.args == ("m1", "obs1")


def test_an_id_that_is_not_in_the_workspace_is_ignored():
    """W に無い id は捨てる。判定は12桁の形を返すが、実在までは見ていない。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _judge(a, "deadbeefdead")
    _run(a, utterance="こんにちは")
    a._memory.record_succession.assert_not_called()


def test_naming_nothing_leaves_the_turn_as_a_root():
    """判定が続き先を返さなければ根になる。「続きではない」がそのまま残る。"""
    a = _agent(
        stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はじめまして"})])]
    )
    _judge(a, None)
    _run(a, utterance="はじめまして")
    a._memory.record_succession.assert_not_called()


def test_a_turn_does_not_follow_itself():
    """自分の起点を指しても繋がない。自己ループはさかのぼりが止まらなくなる。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _judge(a, "obs1")

    async def scenario():
        ip = InformationProcessing(a)
        await ip.push_utterance("こんにちは")

    asyncio.run(scenario())
    a._memory.record_succession.assert_not_called()


def test_a_failing_judge_does_not_stop_the_turn():
    """判定が落ちても、そのターンは成立する。繋がない側へ倒す。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(side_effect=RuntimeError("軽量LLM が落ちた"))
    assert _run(a, utterance="ねえ") == "うん"
    a._memory.record_succession.assert_not_called()
