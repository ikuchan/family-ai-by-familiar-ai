"""ターンどうしを継起でつなぐ（段 3）。

会話は関係の連なりをたどる経路になる（`設計方針_MI間の関係`）。何歩さかのぼるかで窓の
大きさが決まるので、数秒の追跡と数か月の想起が同じ仕組みで出る。

前後は時刻で決めない。会話要約は背景で二秒遅れて書かれるので、時刻順に並べるとやりとりが
入れ違う。辺そのものが順序を持つ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _turn


def _run_turns(a, *utterances):
    """一つの器で続けて反復する。

    実物の `InformationProcessing` はエージェントに一つで、ターンをまたいで生きる
    （`agent.py` が `_info_processing` を持ち回す）。ターンごとに作り直すと、持ち越しが
    毎回消えて再起動と同じ状態になり、連なりの検査にならない。
    """

    async def scenario():
        ip = InformationProcessing(a)
        for u in utterances:
            await ip.run_iteration(u)

    asyncio.run(scenario())


def test_the_second_turn_is_linked_to_the_first():
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="t", name="say", input={"text": "うん"})]),
            _turn([ToolCall(id="t2", name="say", input={"text": "はい"})]),
        ]
    )
    a._memory.latest_exchange_origin = MagicMock(return_value=None)
    _run_turns(a, "ひとつめ", "ふたつめ")
    # obs1＝一つめの起点、obs3＝ふたつめの起点（obs2 は一つめの答え）。
    assert a._memory.record_succession.call_args.args == ("obs1", "obs3")
    # 一つめには前が無いので、辺は一本だけ。二本あれば自分自身か空を結んでいる。
    assert a._memory.record_succession.call_count == 1


def test_the_first_turn_after_a_restart_picks_up_the_chain():
    """持ち越しは再起動で消える。消えたら DB から前の起点を引き直す。

    引き直しが無ければ、再起動のたびに連なりが切れて段 4 で直近を辿れない。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "うん"})])])
    a._memory.latest_exchange_origin = MagicMock(return_value="前回の起点")
    _run_turns(a, "再起動後のひとつめ")
    assert a._memory.record_succession.call_args.args == ("前回の起点", "obs1")


def test_the_lookup_happens_only_once():
    """引き直しは起動後の一度だけ。毎ターン引くと、ターンごとに無駄な問い合わせが増える。"""
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="t", name="say", input={"text": "うん"})]),
            _turn([ToolCall(id="t2", name="say", input={"text": "はい"})]),
        ]
    )
    a._memory.latest_exchange_origin = MagicMock(return_value=None)
    _run_turns(a, "ひとつめ", "ふたつめ")
    assert a._memory.latest_exchange_origin.call_count == 1
