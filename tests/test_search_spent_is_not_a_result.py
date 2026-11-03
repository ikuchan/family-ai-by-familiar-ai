"""出-dh 3 回目のメモ（2026-11-03・本人の決定ア・ア・ア）。

1. 使い切って投げなかった 3 回目の調べものを、「結果が届いた」と書かない（調べていないのに、検査や主LLM が
   結果の 1 件として読む）。器は完了の行き先なので残し、書き方だけ変える。
2. 使い切ったあとに、まだ届いていなかった調べものが失敗して届いたら、ほかに結果の届いた調べものがあるとき
   だけ「使い切った」と同じに分ける（1 回目の結果を、軽く伝えるだけで終わらせない）。
3. 主LLM への決まりの文に、調べものは 1 つの求めで合わせて 2 回までと書く。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core.completion_kind import SEARCH_SPENT, kind_of
from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import Lookup
from tests._arbiter_fakes import decide
from tests.test_arbiter_utterance_by_meaning import _jev
from tests.test_search_twice_then_answer import _ip

_SPENT_TEXT = f"{SEARCH_SPENT}。届いた結果から、分かったことと分からないことを伝える"


def _with_spent(ip):
    ip._req.lookups += [
        Lookup(index=1, action="search_deferred", query="語1", generation=0, result="結果1"),
        Lookup(index=2, action="search_deferred", query="語2", generation=0, result="結果2"),
        Lookup(index=3, action="search_deferred", query="語3", generation=0, result=_SPENT_TEXT),
    ]
    return ip


# ── 1. 使い切りの器を「届いた」と書かない ─────────────────────────────


def test_the_spent_lookup_is_written_as_not_thrown():
    lines = workspace.lookup_lines(_with_spent(_ip())._req)
    assert lines[2] == "3番：search_deferred「語3」は使い切ったので投げなかった"
    assert all(SEARCH_SPENT not in ln for ln in lines)


def test_real_results_are_still_written_as_arrived():
    """**反証の側**：使い切りでない器は、いままでどおり「結果が届いた」。"""
    lines = workspace.lookup_lines(_with_spent(_ip())._req)
    assert lines[0] == "1番：search_deferred「語1」の結果が届いた：結果1"
    assert lines[1] == "2番：search_deferred「語2」の結果が届いた：結果2"


# ── 2. 深く調べる材料に入れない ─────────────────────────────────────


def test_deep_research_material_leaves_out_the_spent_lookup():
    got = _with_spent(_ip())._search_results()
    assert got == "- 語1：結果1\n- 語2：結果2"


# ── 3. 使い切ったあとの失敗 ─────────────────────────────────────────


def test_a_late_failure_after_spending_answers_from_the_results():
    got = kind_of(
        "search_deferred",
        failed=True,
        result="道具が使えず失敗した",
        origin="発話",
        spent_with_results=True,
    )
    assert got is not None and got[1] == ("reply_full",)


def test_a_self_started_late_failure_does_not_answer_unasked():
    got = kind_of(
        "search_deferred",
        failed=True,
        result="道具が使えず失敗した",
        origin="情動",
        spent_with_results=True,
    )
    assert got is not None and "reply_full" not in got[1]


def test_a_plain_failure_is_still_told_lightly():
    """**反証の側**：条件が無ければ、失敗は軽く伝えるのまま。"""
    got = kind_of("search_deferred", failed=True, result="道具が使えず失敗した", origin="発話")
    assert got is not None and got[1] == ("tell_light",)


# ── 4. 調停から使える ──────────────────────────────────────────────


def _flag_passed(**kw) -> bool:
    """調停が完了を分けるとき、`kind_of` に渡した「使い切っていて結果がある」の値。

    分岐（light／full）では見分けられない：試しの書き手は軽い返事の文を返さないので、どれも full に落ちる。
    """
    from unittest.mock import patch

    from familiar_agent.core import completion_kind

    seen: list = []
    real = completion_kind.kind_of

    def spy(*a, **k):
        seen.append(k.get("spent_with_results"))
        return real(*a, **k)

    with patch.object(completion_kind, "kind_of", spy):
        asyncio.run(
            decide(
                jev=_jev({}),
                origin="発話",
                utterance="ベスト11を調べて",
                returned=(("search_deferred", True, "道具が使えず失敗した"),),
                **kw,
            )
        )
    assert seen, "調停が kind_of を呼ばなかった"
    return bool(seen[-1])


def test_the_arbiter_passes_spent_with_results():
    assert _flag_passed(spent_actions=frozenset({"search_deferred"}), search_arrived=True) is True


def test_the_arbiter_needs_both_conditions():
    """**反証の側**：使い切っただけ・結果があるだけでは渡さない（失敗は軽く伝えるのまま）。"""
    assert _flag_passed(spent_actions=frozenset({"search_deferred"})) is False
    assert _flag_passed(search_arrived=True) is False


def test_the_loop_counts_arrived_searches_without_the_spent_lookup():
    """求めから数える：使い切りの器と失敗は、結果の届いた調べものに数えない。"""
    from familiar_agent.loop.event_loop import _search_arrived

    ip = _ip()
    ip._req.lookups += [
        Lookup(
            index=1, action="search_deferred", query="語1", generation=0, result="x", failed=True
        ),
        Lookup(index=2, action="search_deferred", query="語2", generation=0, result=_SPENT_TEXT),
    ]
    assert _search_arrived(ip._req) is False
    ip._req.lookups.append(
        Lookup(index=3, action="search_deferred", query="語3", generation=0, result="結果3")
    )
    assert _search_arrived(ip._req) is True


# ── 5. 決まりの文 ─────────────────────────────────────────────────


def test_the_rule_says_two_searches_per_request():
    from familiar_agent.loop import prompt

    text = prompt.EVENT_SYSTEM_PROMPT
    start = text.index(":id widen-before-giving-up")
    rule = text[start : text.index("(constraint", start)]
    assert "1 つの求めで合わせて 2 回まで" in rule
    assert "語を大きく変えて" in rule


def test_the_spent_completion_is_not_written_into_the_window_talk():
    """窓の会話にも「調べた（…）：使い切った…」と書かない（調べていない）。"""
    import asyncio as _asyncio

    from familiar_agent.loop.event_loop import Trigger

    ip = _ip()
    ip._req.trigger_kind = "発話"
    lines: list = []
    ip._talk_line = lambda who, text: lines.append(text)
    ip._req.lookups.append(Lookup(index=3, action="search_deferred", query="語3", generation=0))
    ip._drained_completions.append(Trigger(kind="完了", query="語3", result=_SPENT_TEXT, index=3))

    async def scenario():
        await ip._intake()
        await ip.close()

    _asyncio.run(scenario())
    assert not any(SEARCH_SPENT in ln for ln in lines), lines


def test_a_real_result_is_still_written_into_the_window_talk():
    """**反証の側**：本当に届いた結果は、窓の会話に 1 行で足す。"""
    import asyncio as _asyncio

    from familiar_agent.loop.event_loop import Trigger

    ip = _ip()
    ip._req.trigger_kind = "発話"
    lines: list = []
    ip._talk_line = lambda who, text: lines.append(text)
    ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="語1", generation=0))
    ip._drained_completions.append(Trigger(kind="完了", query="語1", result="結果1", index=1))

    async def scenario():
        await ip._intake()
        await ip.close()

    _asyncio.run(scenario())
    assert any("結果1" in ln for ln in lines), lines


def test_an_empty_result_does_not_count_as_arrived():
    """0 件（空の結果）は、答えに使える結果ではない。"""
    from familiar_agent.loop.event_loop import _search_arrived

    ip = _ip()
    ip._req.lookups.append(
        Lookup(index=1, action="search_deferred", query="語1", generation=0, result="")
    )
    assert _search_arrived(ip._req) is False
