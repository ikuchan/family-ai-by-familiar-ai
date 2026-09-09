"""主LLM の決定を、**器のまま渡す**（環-e-に・に-5-い）。

`Decision` は環-h で作った器で、主LLM の返りと「投げたときに見ていたもの」を1つに束ねて
いる。ところが `_iterate` は、その器を受け取っておきながら **11 行かけて 12 個の引数へ
ばらしてから** `_act_on_decision` へ渡していた。器を作った意味が呼び口で消えている。

束ねたものを、束ねたまま渡す。**挙動は変えない。**
"""

from __future__ import annotations

import inspect

from familiar_agent.loop.event_loop import Decision, InformationProcessing


def test_the_action_takes_the_decision_itself():
    """受け取るのは器1つと、器が持っていないもの（誰の言葉か・どの世代か）だけ。"""
    params = list(inspect.signature(InformationProcessing._act_on_decision).parameters)
    assert params == ["self", "decision", "utterance", "gen"]


def test_the_decision_carries_everything_the_action_needs():
    """器に無い欄を引数で足していない（足せば、また器の外に事実が散る）。"""
    fields = set(Decision.__dataclass_fields__)
    assert {
        "result",
        "memories",
        "w_id_map",
        "recent_ctx",
        "system",
        "effort",
        "capped",
        "retried",
        "original_text",
    } <= fields


def test_the_iteration_hands_the_decision_over_without_unpacking():
    """呼び口で器をばらさない。ばらせば、欄が増えるたびに呼び口も伸びる。"""
    src = inspect.getsource(InformationProcessing._iterate)
    call = src[src.index("_act_on_decision(") :]
    call = call[: call.index(")") + 1]
    assert "decided." not in call, f"呼び口で器をばらしている：{call}"


def test_the_iteration_got_shorter():
    """に-5-い の目的は行数である。**数字で押さえる**（環-h では逆に増えた）。"""
    src = inspect.getsource(InformationProcessing._iterate)
    assert len(src.split("\n")) <= 236, "薄くなっていない"
