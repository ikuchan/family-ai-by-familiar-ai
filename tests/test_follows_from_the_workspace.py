"""そのターンが何に続くかを、W の中から名指させる（段 4）。

**続き先は、その記録を作るのに使ったものの中にしかない。** W に並んでいるのは、そのとき
頭にあったものである。そこから一つ選ばせるのが、生成のパターンを残すということである
（`設計方針_MI間の関係`）。

id の空間は `memory_verdicts` と同じで、突き合わせも同じ対応表（`_w_index`）を通す。
前方一致で当てずっぽうに引くと、写し間違いが黙って別の記憶へ適用される。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _run, _turn


def test_the_named_memory_becomes_what_this_turn_follows():
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "うん", "follows": "m1"})]),
        ]
    )
    _run(a, utterance="さっきの話だけど")
    # 前＝名指された記憶、後＝このターンの起点。
    assert a._memory.record_succession.call_args.args == ("m1", "obs1")


def test_an_id_that_is_not_in_the_workspace_is_ignored():
    """W に無い id は無視する。**写し間違いが黙って別の記憶へ繋がるのを防ぐ。**"""
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "うん", "follows": "でたらめ"})]),
        ]
    )
    _run(a, utterance="こんにちは")
    a._memory.record_succession.assert_not_called()


def test_naming_nothing_leaves_the_turn_as_a_root():
    """何も名指さなければ根になる。「そのときは思い出していなかった」がそのまま残る。"""
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "はじめまして"})]),
        ]
    )
    _run(a, utterance="はじめまして")
    a._memory.record_succession.assert_not_called()


def test_a_turn_does_not_follow_itself():
    """自分の起点を名指しても繋がない。自己ループはさかのぼりが止まらなくなる。"""
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "うん", "follows": "obs1"})]),
        ]
    )

    async def scenario():
        ip = InformationProcessing(a)
        ip._w_index = {"obs1": "obs1"}
        await ip.run_iteration("こんにちは")

    asyncio.run(scenario())
    a._memory.record_succession.assert_not_called()
