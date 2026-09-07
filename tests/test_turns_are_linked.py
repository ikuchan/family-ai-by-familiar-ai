"""ターンは自動では繋がらない（段 4）。

段 3 では、前の起点があれば無条件に継起の辺を張っていた。それは鎖の種類を機構の側で
数え上げることになり、並行して走る本数に上限が生まれた（会話と見回りが同じ一本に混ざる）。

**続き先は、そのターンを作るのに使った W の中にしかない。** 主LLM が `follows` で名指した
ときだけ繋がる（`設計方針_MI間の関係`）。ここでは、撤去した無条件の連結が戻っていない
ことを見る。名指したときに繋がることは `test_follows_from_the_workspace.py` が見ている。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _turn


def _run_turns(a, *utterances):
    """一つの器で続けて反復する（実物の器はターンをまたいで生きる）。"""

    async def scenario():
        ip = InformationProcessing(a)
        for u in utterances:
            await ip.run_iteration(u)

    asyncio.run(scenario())


def test_two_turns_are_not_linked_on_their_own():
    """名指さなければ、続けて話しても繋がらない。

    繋がってしまうと、夜中の見回りが朝と夜の会話のあいだに割り込む。
    """
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="t", name="say", input={"text": "うん"})]),
            _turn([ToolCall(id="t2", name="say", input={"text": "はい"})]),
        ]
    )
    _run_turns(a, "ひとつめ", "ふたつめ")
    a._memory.record_succession.assert_not_called()


def test_the_loop_does_not_ask_the_store_for_a_previous_origin():
    """起動時に前の起点を引き直す仕組みは撤去した。

    持ち越しを持たないので、再起動で連なりが切れる問題そのものが無くなった。口も
    落としてあるので、呼んでいれば属性エラーになる。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "うん"})])])
    _run_turns(a, "ひとつめ")
    ip = InformationProcessing(a)
    assert not hasattr(ip, "_last_origin_id")
    assert not hasattr(ip, "_seeded_origin")
