"""**考えた回数に上限を置く**（暴走の歯止め・2026-09-12 実機で露見）。

環-h は「主LLM の返りで反復数を 0 へ戻す」と決めた——主LLM が材料を見て「足りない」と
判断したなら新しい一巡である。あわせて「機械の歯止めは足さず、考えた回数を材料として
渡す」とした。

実機で輪が閉じた。カメラが映像を返さず `see` が空で完了し、主LLM は「目の前を見る」を
出し続けた。同じ語は二度と投げず**完了として積む**ので、その完了が次の反復を起こし、主LLM が
同じ要求を出す。**反復は毎回 1/5 で上限が効かず、考えた回数は 57 まで数えたが上限が無い。**
止めたのは人の手だった。

**「主LLM に判断させる」は否定された。** `考え=57回目` と渡していても止まらなかった。見える
はずのものが見えていない状況で、もう一度見ようとするのは自然な判断で、指示では止まらない。

上限は**考えた回数**に置く（反復ではなく）。反復のリセットは残す。上限に達したら `capped` を
立て、`say` だけを渡して**答えさせて閉じる**——その仕組みは既にある。
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.request import Lookup, Request


def _ip(rounds_done: int):
    ip = object.__new__(InformationProcessing)
    ip._req = Request()
    ip._req.lookups = [
        Lookup(index=i + 1, action="主LLM", query=f"主LLM{i + 1}", generation=0)
        for i in range(rounds_done)
    ]
    ip._agent = MagicMock()
    ip._agent.config.event_max_iterations = 5
    ip._agent.config.max_thinking_rounds = 5
    return ip


def test_the_config_names_the_cap():
    from familiar_agent.config import AgentConfig

    assert "max_thinking_rounds" in AgentConfig.__dataclass_fields__
    assert AgentConfig().max_thinking_rounds == 5


def test_below_the_cap_thinking_is_not_capped():
    ip = _ip(rounds_done=3)  # 4回目
    assert ip._thinking_round == 4
    assert not ip._thinking_capped


def test_at_the_cap_thinking_is_capped():
    """5回考えて答えが出ないなら、6回目で出る見込みは薄い。答えさせて閉じる。"""
    ip = _ip(rounds_done=4)  # 5回目
    assert ip._thinking_round == 5
    assert ip._thinking_capped


def test_the_iteration_caps_on_thinking_rounds_too():
    """`capped` は反復の上限**か**考えた回数の上限で立つ。反復のリセットは残す。"""
    src = inspect.getsource(InformationProcessing._iterate)
    assert "_thinking_capped" in src, "反復が考えた回数の上限を見ていない"
    assert "capped = chain >= max_chain or " in src or "or self._thinking_capped" in src


def test_the_cap_is_logged_as_a_brake():
    """上限で打ち切ったことは、後からログだけで判別できる必要がある。"""
    src = inspect.getsource(InformationProcessing._iterate)
    assert "考えた回数" in src and "打ち切る" in src
