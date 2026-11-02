"""調べもの 2 回を使い切ったら、パジュから問い、はいなら深く調べる（出-dh 段 2・出-dg ろ・2026-11-02・本人の決定ア）。

ふつうの調べものを 2 回使い切っても答えが足りないとき、主LLM の答えを話して書き残したあと、機械が決まった問い「もっと詳しく、
時間をかけて調べましょうか？」を話す。預かり方は、タイマーと同じ確認待ち（5 分のあいだ、どの言葉も 3 択でしか読まず、タイマー
も止める）ではなく、申し出の預かり（出-dj）にする。次の 1 回の返事だけを、同じ窓のあいだだけ 3 択で読み、はいなら預かった
元の頼みと、前の求めで届いた調べものの結果で深く調べる。それ以外はふつうに読む。本物の API は呼ばない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.loop.event_loop import Lookup
from familiar_agent.store import arbiter_records
from familiar_agent.tools import deep_research as dr
from familiar_agent.tools.deep_research import OFFER_TEXT
from tests._arbiter_fakes import decide, ws_of
from tests.test_arbiter_utterance_by_meaning import _a, _jev
from tests.test_deep_research import _ip

REQUEST = "子どもの読書の習慣のつけ方を調べて"


def test_the_question_is_the_decided_sentence():
    assert OFFER_TEXT == "もっと詳しく、時間をかけて調べましょうか？"


# ── 問う材料と、問い ─────────────────────────────────────────────────────


def _spent(ip):
    ip._req.request_text = REQUEST
    ip._req.spent_actions.add("search_deferred")
    ip._req.lookups += [
        Lookup(index=1, action="search_deferred", query="語1", generation=0, result="一回目の結果"),
        Lookup(index=2, action="search_deferred", query="語2", generation=0, result="二回目の結果"),
    ]
    return ip


def test_the_material_holds_the_request_and_what_was_found():
    """後始末が器を片づける前に、元の頼みと届いた結果を控える（はいの求めには前の結果が無い）。"""
    deep = _spent(_ip())._deep_offer_material()
    assert deep is not None and deep["request"] == REQUEST
    assert "一回目の結果" in deep["found"] and "二回目の結果" in deep["found"]


@pytest.mark.parametrize(
    "setup",
    [
        lambda ip: ip._req.spent_actions.discard("search_deferred"),  # 使い切っていない
        lambda ip: ip._req.spent_actions.add("deep_research"),  # もう深く調べた
        lambda ip: setattr(
            ip._agent, "confirm_alive", MagicMock(return_value=True)
        ),  # 確認待ちがある
    ],
)
def test_no_question_when_it_does_not_fit(setup):
    """**反証の側**：使い切っていない・もう深く調べた・確認待ちがある、なら問わない。"""
    ip = _spent(_ip())
    setup(ip)
    assert ip._deep_offer_material() is None


def test_no_question_when_deep_research_is_not_available():
    """**反証の側**：主LLM の担い手が Anthropic でなければ問わない。"""
    assert _spent(_ip("gemini"))._deep_offer_material() is None


def test_the_question_is_spoken_and_held_as_an_offer_not_a_confirm():
    ip = _ip()
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    deep = {"request": REQUEST, "found": "- 語1：一回目の結果"}

    async def go():
        await ip._offer_deep(deep)
        await ip.close()

    asyncio.run(go())
    ip._say_aloud.assert_awaited_once_with(OFFER_TEXT)
    assert ip._offer["question"] == OFFER_TEXT and ip._offer["deep"] == deep
    ip._agent.ask_confirm.assert_not_called()  # **反証の側**：確認待ち（5 分 3 択だけ）にはしない


def test_no_question_when_paju_may_not_speak():
    """**反証の側**：話してよい門が閉じていれば、問わず、預けもしない。"""
    ip = _ip()
    ip._delivery_block_reason = lambda: "静穏"  # type: ignore[method-assign]
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]

    async def go():
        await ip._offer_deep({"request": REQUEST, "found": ""})
        await ip.close()

    asyncio.run(go())
    ip._say_aloud.assert_not_awaited()
    assert ip._offer is None


# ── はいで深く調べる ──────────────────────────────────────────────────────


def _answer(monkeypatch, pick: str, offer_deep=None):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"answer": _a(pick, 0.9)}, {"meaning": _a("other", 0.9)})
    return asyncio.run(
        decide(
            jev=jev,
            utterance="はい",
            origin="発話",
            offer_question=OFFER_TEXT,
            offer_deep=offer_deep,
        )
    )


def test_yes_throws_deep_research_with_what_was_held(monkeypatch):
    deep = {"request": REQUEST, "found": "- 語1：一回目の結果"}
    d = _answer(monkeypatch, "yes", deep)
    assert (d.branch, d.action, d.query) == ("action", "deep_research", REQUEST)
    assert d.tool_input == {**deep, "query": REQUEST}  # 器の見出しが頼みになる（`query`）


def test_a_plain_offer_still_goes_to_the_main_llm(monkeypatch):
    """**反証の側**：深く調べるでない申し出（出-dj）の「はい」は、いまどおり主LLM へ。"""
    d = _answer(monkeypatch, "yes")
    assert (d.branch, d.offer_accepted) == ("full", True)


def test_other_words_are_read_as_usual(monkeypatch):
    """**反証の側**：それ以外の言葉は、ふつうに読む（確認待ちのように止めない）。"""
    d = _answer(monkeypatch, "other", {"request": REQUEST, "found": ""})
    assert d.action != "deep_research"


def test_the_loop_hands_the_held_input_to_the_arbiter():
    ip = _ip()
    wake = ip._wake_window()
    wake.open(time.monotonic())
    ip._req.window_session = wake.session
    deep = {"request": REQUEST, "found": ""}
    ip._offer = {"question": OFFER_TEXT, "session": wake.session, "deep": deep}
    seen: dict = {}

    async def fake_decide(self, inp, on_decided=None):
        from familiar_agent.loop import arbiter

        seen["inp"] = inp
        return arbiter.Decision(branch="light", text="うん")

    with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
        asyncio.run(
            ip._decide(utterance="はい", ws=ws_of(), present_ctx="", capped=False, round_=1)
        )
    assert seen["inp"].offer_question == OFFER_TEXT and seen["inp"].offer_deep == deep


def test_the_deep_model_gets_the_held_request_and_results():
    ip = _ip()
    ip._req.request_text = "はい"
    seen: dict = {}

    async def fake(client, **kw):
        seen.update(kw)
        return dr.Answer(text="答え。", failed=False, why="", seconds=1.0)

    async def go():
        ip._req.lookups.append(Lookup(index=1, action="deep_research", query="q", generation=0))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dr, "research", fake)
            await ip._run_lookup_body(
                "deep_research", {"request": REQUEST, "found": "- 語1：一回目の結果"}, "q", None, 1
            )
        await ip.close()

    asyncio.run(go())
    assert f"頼まれたこと：{REQUEST}" in seen["user"]  # 「はい」ではない
    assert "一回目の結果" in seen["user"]  # 前の求めの調べものの結果を渡す


# ── 見直しで足した確かめ（作り直しの 2 回目）─────────────────────────────────────


def test_no_question_after_an_answer_that_already_offered():
    """**反証の側**：答えが申し出で終わっていたら（出-dj の預かりがある）、問いを 2 つ続けない。"""
    ip = _spent(_ip())
    ip._offer = {"question": "一覧にしましょうか？", "session": 1}
    assert ip._deep_offer_material() is None


def test_no_question_without_a_request():
    """**反証の側**：頼みが空なら、何を深く調べるかが無いので問わない（「はい」を調べない）。"""
    ip = _spent(_ip())
    ip._req.request_text = ""
    assert ip._deep_offer_material() is None


def test_an_interrupted_question_is_not_held():
    """**反証の側**：問いを話しているあいだに次の求めが始まったら、古い問いを預けない。"""
    ip = _ip()
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    gen = ip._request_generation

    async def say(text, **kw):
        ip._request_generation += 1  # 話しているあいだに打ち切られた

    ip._say_aloud = say  # type: ignore[method-assign]

    async def go():
        await ip._offer_deep({"request": REQUEST, "found": ""}, gen=gen)
        await ip.close()

    asyncio.run(go())
    assert ip._offer is None


def test_held_empty_results_are_not_replaced_by_this_requests():
    """預けた結果が空なら空のまま渡す（「はい」の求めの調べものを、前の求めの結果として渡さない）。"""
    ip = _ip()
    ip._req.lookups.append(
        Lookup(
            index=1, action="search_deferred", query="別", generation=0, result="いまの求めの結果"
        )
    )
    seen: dict = {}

    async def fake(client, **kw):
        seen.update(kw)
        return dr.Answer(text="答え。", failed=False, why="", seconds=1.0)

    async def go():
        ip._req.lookups.append(Lookup(index=2, action="deep_research", query="q", generation=0))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dr, "research", fake)
            await ip._run_lookup_body(
                "deep_research", {"request": REQUEST, "found": ""}, "q", None, 2
            )
        await ip.close()

    asyncio.run(go())
    assert "いまの求めの結果" not in seen["user"]
