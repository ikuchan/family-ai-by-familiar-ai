"""調べものは合わせて 2 回まで。2 回目の結果が届いたら機械が `search_deferred` を外す（出-dg ろ・2026-11-01・本人の決定ア）。

10/31 21:51、主LLM は 1 回調べただけで「探し直しましょうか」と聞いて終えた。決まり（`widen-before-giving-up`）で語を変えて
1 回探し直させ、3 回目は言葉でなく機械が止める（出-o と同じ形・禁じたい動作は機械が落とす）。話す前の検査は、記録に
書いた版（8,192 字で切る）でなく、W と同じく手元から組み直した版を事実にする（2 回目の結果を見られるように）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger
from tests._arbiter_fakes import decide
from tests.test_arbiter_utterance_by_meaning import _a, _jev
from tests.test_event_loop import _agent
from tests.test_workspace_is_the_core import _rec


def _ip():
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.request_id = "req-1"
    return ip


def _returns(ip, *done: "tuple[int, bool]") -> None:
    """調べものの結果を届ける（(index, 失敗か)）。取込は完了箱から受ける（環-m）。"""
    for index, failed in done:
        ip._req.lookups.append(
            Lookup(index=index, action="search_deferred", query=f"語{index}", generation=0)
        )
        ip._drained_completions.append(
            Trigger(
                kind="完了", query=f"語{index}", result=f"結果{index}", index=index, failed=failed
            )
        )

    async def scenario():
        await ip._intake()
        await ip.close()

    asyncio.run(scenario())


# ── 2 回届いたら外す ──────────────────────────────────────────────────


def test_two_results_spend_the_search():
    ip = _ip()
    _returns(ip, (1, False), (2, False))
    assert "search_deferred" in ip._req.spent_actions


def test_one_result_does_not():
    """**反証の側**：1 回目のあとは、探し直せる。"""
    ip = _ip()
    _returns(ip, (1, False))
    assert "search_deferred" not in ip._req.spent_actions


def test_a_failed_search_is_not_counted():
    """**反証の側**：道具が使えなかった回は結果ではない（出-o）。"""
    ip = _ip()
    _returns(ip, (1, True), (2, False))
    assert "search_deferred" not in ip._req.spent_actions


def test_the_main_llm_loses_the_search_tool():
    ip = _ip()
    before = {d["name"] for d in ip._tools(actions=("say", "search_deferred"))}
    ip._req.spent_actions.add("search_deferred")
    after = {d["name"] for d in ip._tools(actions=("say", "search_deferred"))}
    assert "search_deferred" in before and "search_deferred" not in after


def test_the_arbiter_does_not_offer_a_third_search():
    seen: list = []
    jev = _jev({"action": _a("reply_full", 0.9)})
    real = jev.ask

    async def ask(state, questions):
        seen.append(questions)
        return await real(state, questions)

    jev.ask = ask
    kw = {"returned": (("search_deferred", False, "結果2"),), "utterance": "ベスト11を調べて"}
    asyncio.run(decide(jev=jev, origin="発話", spent_actions=frozenset({"search_deferred"}), **kw))
    assert "search_deferred" not in seen[-1]["action"]["criteria"]
    seen.clear()
    asyncio.run(decide(jev=jev, origin="発話", **kw))  # 反証の側：使い切っていなければ並ぶ
    assert "search_deferred" in seen[-1]["action"]["criteria"]


def test_the_spent_set_is_cleared_when_the_request_closes():
    ip = _ip()
    ip._req.spent_actions.add("search_deferred")
    asyncio.run(ip._abort_lookups())
    assert ip._req.spent_actions == set()


# ── 話す前の検査は、組み直した版を事実にする ─────────────────────────────────────

SECOND = "二回目の結果。" * 1100  # 約 7,700 字


def test_the_speech_check_sees_the_second_result_in_full():
    ip = _ip()
    ip._req.request_text = "ベスト11を調べて"
    ip._req.lookups += [
        Lookup(
            index=1,
            action="search_deferred",
            query="語1",
            generation=0,
            result="一回目の結果。" * 1100,
        ),
        Lookup(index=2, action="search_deferred", query="語2", generation=0, result=SECOND),
    ]
    ip._req.live_version_id = "v-live"
    stored = workspace.version_text(ip._req)[:8192]  # 記録に書いた版
    other = _rec("m-old", "前の日の記録", fit=0.4)
    facts = ip._checker_facts([_rec("v-live", stored, fit=0.95), other])
    assert facts.count("二回目の結果。") == 1100


def test_each_search_is_its_own_fact():
    """版は調べもの 1 件ずつに分けて渡す（1 件 10,000 字まで・縮めるときは長い件だけ・出-dg ろ）。"""
    ip = _ip()
    ip._req.request_text = "ベスト11を調べて"
    ip._req.lookups += [
        Lookup(index=1, action="search_deferred", query="語1", generation=0, result="一回目。"),
        Lookup(index=2, action="search_deferred", query="語2", generation=0, result="二回目。"),
    ]
    ip._req.live_version_id = "v-live"
    facts = ip._checker_facts([_rec("v-live", "記録に書いた版", fit=0.95)])
    assert "（2件・これも根拠）" in facts  # 2 件（版の頭の頼みは結果に数えない・知-bn）
    assert "\n  - 1番：" in facts and "\n  - 2番：" in facts


def test_other_records_keep_their_stored_text():
    """**反証の側**：いまの版でない記録は、組み直さない（頼みの記録は結果に入れないので、見た記録で見る・知-bn）。"""
    ip = _ip()
    ip._req.turn_records = [("m-seen", "見た")]  # この求めで見た記録（open）
    facts = ip._checker_facts([_rec("m-seen", "記録に書いた文", fit=0.9)])
    assert "記録に書いた文" in facts


# ── 投げる口でも 2 回で止める（同時に 2 つ・文で書いた呼び出し・調停が選んだ調べものも通る口）────────────


def _dispatch(ip, *done: "tuple[int, bool | None]") -> list:
    """(index, 失敗か) の調べもの（None は飛行中）を置き、3 つ目を投げる。投げた調べものの語を返す。"""
    for index, failed in done:
        ip._req.lookups.append(
            Lookup(
                index=index,
                action="search_deferred",
                query=f"語{index}",
                generation=0,
                result=None if failed is None else f"結果{index}",
                failed=bool(failed),
            )
        )
    thrown: list = []

    async def run(action, tool_input, query, intent_id, index=0):
        thrown.append(query)

    ip._run_lookup = run

    async def scenario():
        ip._dispatch_lookup("search_deferred", {"query": "語9"}, "語9", None)
        await asyncio.sleep(0)
        await ip.close()

    asyncio.run(scenario())
    return thrown


def test_a_third_search_is_not_thrown():
    from familiar_agent.core.completion_kind import SEARCH_SPENT

    ip = _ip()
    assert _dispatch(ip, (1, False), (2, False)) == []
    got = ip._triggers.get_nowait()
    assert got.kind == "完了" and SEARCH_SPENT in got.result
    assert "search_deferred" in ip._req.spent_actions


def test_while_the_second_is_in_flight_it_waits_for_it():
    """2 回目がまだ飛んでいても 3 回目は止める。完了は積まない（2 回目が届いた反復で答える。積むと、その完了で
    答えるか黙って閉じ、2 回目の結果の行き先が無くなる）。"""
    ip = _ip()
    assert _dispatch(ip, (1, False), (2, None)) == []
    assert ip._triggers.empty()
    assert "search_deferred" in ip._req.spent_actions


def test_a_failed_search_leaves_room_for_another():
    """**反証の側**：使えなかった回は数えない（出-o）。"""
    ip = _ip()
    assert _dispatch(ip, (1, False), (2, True)) == ["語9"]


def test_the_spent_completion_asks_the_main_llm_to_answer():
    from familiar_agent.core.completion_kind import SEARCH_SPENT, kind_of

    got = kind_of("search_deferred", failed=False, result=f"{SEARCH_SPENT}。", origin="発話")
    assert got is not None and got[1] == ("reply_full",)


def test_a_self_started_search_does_not_answer_unasked():
    """**反証の側**：自分から調べた（情動）なら、頼まれていないので考えて返さない。"""
    from familiar_agent.core.completion_kind import SEARCH_SPENT, kind_of

    got = kind_of("search_deferred", failed=False, result=f"{SEARCH_SPENT}。", origin="情動")
    assert got is not None and "reply_full" not in got[1]


def test_two_refused_searches_at_once_post_one_completion():
    """同じ返事で 3 回目と 4 回目を出しても、使い切りの完了は 1 つ（2 つ目は 1 つ目が届くのを待つ）。"""
    ip = _ip()
    _dispatch(ip, (1, False), (2, False))
    ip._dispatch_lookup("search_deferred", {"query": "語10"}, "語10", None)
    ip._triggers.get_nowait()
    assert ip._triggers.empty()
