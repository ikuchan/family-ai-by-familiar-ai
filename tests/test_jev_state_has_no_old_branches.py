"""Jev に渡す状態の文から、古い分岐の説明を外す（出-ay 段 5c・2026-10-10・本人の決定ア）。

段 4 の新しい問い（発話の意味と動作・完了と情動の 2 回目）の選択肢に light・full・action は無いのに、状態の文には古い分岐の
先導文（「次のどれかを選ぶ」・「"light" で」「"action"」）・`[判断の目安]`（迷ったら full・材料が無い → action）・上限の
注意（"action" は選べない）が入ったままだった。起点ごとに「いま何が起きたか」を言う先導文に書き直し、いま・顔ぶれ・
言葉・作業状態を残す。軽量LLM の先導文（書くときの指示）は変えない。
"""

from __future__ import annotations

import pytest

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

_OLD = (
    "[判断の目安]",
    "次のどれかを選ぶ",
    '"light"',
    '"full"',
    '"action"',
    "light",
    "full",
    "action",
)


def _state(**kw) -> str:
    arbiter = Arbiter(jev=None, writer=None)
    return arbiter._state(ArbiterInput(utterance="明日の天気は？", workspace_ctx="（作業）", **kw))


@pytest.mark.parametrize(
    "kw",
    [
        {"origin": "発話"},
        {"origin": "発話", "returned": (("search_deferred", False, "晴れ"),)},
        {"origin": "情動", "fired_axis": "safety"},
        {"origin": "機器"},
        {"origin": "発話", "capped": True, "thinking_round": 3},
        {"origin": "発話", "can_see": True},
    ],
)
def test_the_state_has_no_old_branch_words(kw):
    state = _state(**kw)
    for word in _OLD:
        assert word not in state, word
    assert "[いま]" in state and "（作業）" in state and "明日の天気は？" in state


def test_each_origin_says_what_happened():
    assert "人から言葉が届いた" in _state(origin="発話")
    assert "自分の動作の結果が届いた" in _state(
        origin="発話", returned=(("search_deferred", False, "晴れ"),)
    )
    assert "自分の中から湧いた" in _state(origin="情動", fired_axis="safety")
    assert "機器から知らせが届いた" in _state(origin="機器")


def test_the_cap_and_round_notes_stay_without_branch_words():
    state = _state(origin="発話", capped=True, thinking_round=3)
    assert "これ以上は調べられない" in state and "3 回目" in state
