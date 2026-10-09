"""情動は発火した軸で決め、要るときだけ Jev に 2 回目を聞く（出-ay 段 4-3・2026-10-09・`設計方針_判定の段` §2.2.5）。

seeking は調べに行くだけ（Jev に聞かない）。safety は見るか調べるか、bond は軽く／考えて話しかけるか、esteem はそれに
調べるを足して Jev に聞く。確信度は使わない（本人）。結果が届いた反復は完了の道（段 4-2）を通る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from tests._arbiter_fakes import decide, jev_says, writer_says


def _jev_picks(choice: str, confidence: float = 0.1) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True, answers={"action": {"choice": choice, "confidence": confidence}}
        )
    )
    return jev


def _run(axis, *, jev=None, writer=None, returned=()):
    jev = jev or jev_says("full")
    writer = writer or writer_says(
        {
            "text": "パパ、おかえり",
            "query": "守谷市 今週末 イベント",
            "tool_input": {"direction": "右"},
        }
    )
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance=f"[内的な促し:{axis}] したい",
            origin="情動",
            fired_axis=axis,
            can_see=True,
            talking=axis in ("bond", "esteem"),
            returned=returned,
        )
    )
    return d, jev


def test_seeking_goes_to_search_without_asking_jev():
    d, jev = _run("seeking")
    jev.ask.assert_not_awaited()
    assert (d.branch, d.action, d.query) == ("action", "search_deferred", "守谷市 今週末 イベント")


def test_safety_asks_look_or_search_and_ignores_confidence():
    d, jev = _run("safety", jev=_jev_picks("look"))
    (_state, questions), _ = jev.ask.await_args
    assert set(questions) == {"action"}
    assert set(questions["action"]["criteria"]) == {"look", "search_deferred"}
    assert (d.branch, d.action) == ("action", "look")


def test_bond_talks_lightly_or_thinks():
    d, jev = _run("bond", jev=_jev_picks("talk_light"))
    assert set(jev.ask.await_args.args[1]["action"]["criteria"]) == {"talk_light", "talk_full"}
    assert (d.branch, d.text) == ("light", "パパ、おかえり")
    d, _ = _run("bond", jev=_jev_picks("talk_full"))
    assert d.branch == "full"


def test_esteem_may_also_search():
    d, jev = _run("esteem", jev=_jev_picks("search_deferred"))
    assert set(jev.ask.await_args.args[1]["action"]["criteria"]) == {
        "talk_light",
        "talk_full",
        "search_deferred",
    }
    assert (d.branch, d.action) == ("action", "search_deferred")


def test_a_result_inside_an_affect_request_is_the_completion_road():
    d, jev = _run(
        "seeking", jev=_jev_picks("silent"), returned=(("look", False, "部屋に誰もいない"),)
    )
    criteria = set(jev.ask.await_args.args[1]["action"]["criteria"])
    assert criteria == {"look", "talk_light", "silent"}  # 自分から見に行った結果（完了の表）
