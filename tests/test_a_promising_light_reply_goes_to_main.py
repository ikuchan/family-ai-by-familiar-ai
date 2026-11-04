"""出-dl 原因 2：調べものが動いていないのに、軽い返事が先のことを約束したら、文を捨てて主LLM に考えて返させる。

2026-11-03 19:12、「もう一度調べてみて」への軽い返事が「すぐに確かめて…少々お待ち」と約束し、調べものは立たずに
黙った。書き手への指示 `_NO_PROMISE` があっても出た。書いた文を Jev に「先のことを約束しているか」と聞き、はいの確率が
境目（`AgentConfig.arbiter_promise_min`・0.6）以上なら full へ回す（本人の決定イ）。調べもの中は「調べています」が
正しいので聞かない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput
from familiar_agent.loop.event_loop import Lookup
from tests._arbiter_fakes import decide, writer_says
from tests.test_search_twice_then_answer import _ip

# ── 1. 調停の材料：調べもの中か ───────────────────────────────────────


def _in_flight(*lookups: Lookup) -> bool:
    ip = _ip()
    ip._req.lookups += list(lookups)
    return ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").search_in_flight


def test_a_search_without_a_result_is_in_flight():
    assert _in_flight(Lookup(index=1, action="search_deferred", query="語", generation=0))


def test_a_deep_research_without_a_result_is_in_flight():
    assert _in_flight(Lookup(index=1, action="deep_research", query="語", generation=0))


def test_arrived_searches_and_other_tools_are_not_in_flight():
    """**反証の側**：結果の届いた調べもの・思い出しは、調べもの中ではない。"""
    assert not _in_flight(
        Lookup(index=1, action="search_deferred", query="語", generation=0, result="結果"),
        Lookup(index=2, action="recall", query="語", generation=0),
    )


# ── 2. 調停：約束している軽い返事は主LLM へ ────────────────────────────

PROMISE = "すぐに確かめてお伝えします"


def _jev_says(promise: "float | None", *, ok: bool = True) -> MagicMock:
    """意味は「その他」・動作は「軽く返す」、約束の問いには `promise`（はいの確率）を答える偽の Jev。"""
    answers = {
        "meaning": {"choice": "other", "confidence": 0.9, "probabilities": {"other": 0.9}},
        "action_other": {
            "choice": "reply_light",
            "confidence": 0.9,
            "probabilities": {"reply_light": 0.9},
        },
    }
    if promise is not None:
        answers["promise"] = {"type": "noul", "noul": promise}
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return jev


def _decide(jev, **inp):
    return asyncio.run(
        decide(
            jev=jev, writer=writer_says({"text": PROMISE}), utterance="もう一度調べてみて", **inp
        )
    )


def _asked_promise(jev) -> bool:
    return any("promise" in c.args[1] for c in jev.ask.await_args_list)


def test_a_promising_light_reply_goes_to_the_main_llm():
    jev = _jev_says(0.96)
    d = _decide(jev)
    assert d.branch == "full"
    state, questions = next(c.args for c in jev.ask.await_args_list if "promise" in c.args[1])
    assert PROMISE in state and "もう一度調べてみて" in state


def test_the_boundary_counts_as_a_promise():
    """ちょうど境目（0.6）は約束として扱う（`>=`）。"""
    assert _decide(_jev_says(0.6)).branch == "full"


def test_a_light_reply_without_a_promise_is_kept():
    """**反証の側**：約束していなければ、書いた文をそのまま言う。"""
    d = _decide(_jev_says(0.2))
    assert (d.branch, d.text) == ("light", PROMISE)


def test_just_below_the_boundary_is_kept():
    """**反証の側**：境目のすぐ下（0.59）は約束ではない。"""
    assert _decide(_jev_says(0.59)).branch == "light"


def test_while_searching_the_promise_is_not_asked():
    """**反証の側**：調べもの中は「調べています」が正しいので聞かない。"""
    jev = _jev_says(0.96)
    d = _decide(jev, search_in_flight=True)
    assert (d.branch, d.text) == ("light", PROMISE)
    assert not _asked_promise(jev)


def _fails_on_the_promise(failure) -> MagicMock:
    """意味の問いには答え、約束の問いだけで失敗する偽の Jev（`failure` は返りか例外）。"""
    jev = _jev_says(None)
    meaning = jev.ask.return_value

    async def ask(state, questions):
        if "promise" not in questions:
            return meaning
        if isinstance(failure, Exception):
            raise failure
        return failure

    jev.ask = AsyncMock(side_effect=ask)
    return jev


@pytest.mark.parametrize(
    "make",
    [
        lambda: _jev_says(None),  # 答えに欄が無い
        lambda: _fails_on_the_promise(JevAnswer(ok=False, error="HTTP 500")),
        lambda: _fails_on_the_promise(TimeoutError()),
    ],
)
def test_without_an_answer_the_reply_is_kept(make):
    """**反証の側**：答えが無い・Jev が失敗した・例外が出たら、いまどおり書いた文を言う。"""
    jev = make()
    d = _decide(jev)
    assert _asked_promise(jev)
    assert (d.branch, d.text) == ("light", PROMISE)


def test_an_acting_reply_is_not_asked():
    """**反証の側**：黙る依頼・名乗りの返事（約束を止めない）は聞かない。"""
    jev = _jev_says(0.96)
    arbiter = Arbiter(jev=jev, writer=writer_says({"text": PROMISE}), timeout=2.0)
    d = asyncio.run(
        arbiter._completion_decision(
            ArbiterInput(utterance="x", workspace_ctx=""), "reply_light", tool="", no_promise=False
        )
    )
    assert (d.branch, d.text) == ("light", PROMISE)
    assert not _asked_promise(jev)


def test_an_ask_back_is_not_asked():
    """**反証の側**：聞き返しは、いまの「問いになっているか」の検査だけ。"""
    jev = _jev_says(0.96)
    arbiter = Arbiter(jev=jev, writer=writer_says({"text": "どれのこと？"}), timeout=2.0)
    d = asyncio.run(
        arbiter._completion_decision(
            ArbiterInput(utterance="x", workspace_ctx=""), "ask_back", tool="", no_promise=True
        )
    )
    assert (d.branch, d.text) == ("light", "どれのこと？")
    assert not _asked_promise(jev)


def test_the_boundary_comes_from_the_config():
    from familiar_agent.config import AgentConfig

    assert AgentConfig().arbiter_promise_min == 0.6


def test_the_loop_passes_the_boundary_to_the_arbiter():
    ip = _ip()
    ip._agent.config.arbiter_promise_min = 0.75
    assert ip._arbiter()._promise_min == 0.75
