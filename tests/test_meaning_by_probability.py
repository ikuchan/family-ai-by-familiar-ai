"""意味と動作は確率で決める（出-db 段 A・2026-10-25・本人の決定 1-a・`設計方針_判定の段` §2.2.5）。

確信度 0.6 で絞っていたころは、確率が害の無い選択肢（軽く返す・考えて返す）に割れただけで聞き返しに倒れ、
「パジュ、しりとりしよ」を毎回聞き返した（10/24 18:39〜18:42）。返す側（軽く返す＋考えて返す）の確率の和が、ほかの
どの動作の確率よりも大きければ返す。名乗り・否定は誰かを決めつけるので 0.6 のまま。迷ったときに 2 回目を聞く問いは外す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import utterance_meaning as um
from tests._arbiter_fakes import decide, writer_says


def _p(choice: str, confidence: float, **probs: float) -> dict:
    return {
        "choice": choice,
        "confidence": confidence,
        "probabilities": probs or {choice: confidence},
    }


def test_a_split_between_two_answers_is_answered_not_asked_back():
    """10/24 18:39「パジュ、しりとりしよ」の形：意味 0.5・軽く返す 0.4・考えて返す 0.3・聞き返す 0.3。"""
    got = um.decide(
        _p("other", 0.5, other=0.5, answerable=0.3, unformed=0.2),
        _p("reply_light", 0.4, reply_light=0.4, reply_full=0.3, ask_back=0.3),
    )
    assert got.final == "reply_light"


def test_the_larger_answer_is_used():
    got = um.decide(
        _p("other", 0.7, other=0.7),
        _p("reply_light", 0.35, reply_light=0.35, reply_full=0.4, ask_back=0.25),
    )
    assert got.final == "reply_full"


def test_asking_back_wins_only_over_the_sum_of_the_answers():
    """**反証の側**：聞き返すが返す側の和より大きければ、聞き返す。"""
    got = um.decide(
        _p("other", 0.9, other=0.9),
        _p("ask_back", 0.6, ask_back=0.6, reply_light=0.25, reply_full=0.15),
    )
    assert got.final == "ask_back"


def test_a_tool_is_taken_by_its_probability():
    got = um.decide(
        _p("time", 0.5, time=0.5, music=0.3, other=0.2),
        _p("set_timer", 0.45, set_timer=0.45, ask_back=0.3, cancel_timer=0.25),
    )
    assert got.final == "set_timer"


def test_without_probabilities_the_choice_is_used():
    got = um.decide(
        {"choice": "music", "confidence": 0.5}, {"choice": "play_music", "confidence": 0.4}
    )
    assert got.final == "play_music"


def test_claims_and_denials_still_need_confidence():
    """**反証の側**：名乗り・否定は 0.6 のまま。足りなければ聞き返す（2 回目の問いは無い）。"""
    sure = um.decide(_p("claim", 0.9), _p("パパ", 0.9))
    assert (sure.final, sure.name) == ("claim", "パパ")
    assert um.decide(_p("claim", 0.9), _p("パパ", 0.5)).final == "ask_back"
    assert um.decide(_p("deny", 0.5), _p("ママ", 0.9)).final == "ask_back"


def test_the_second_question_is_gone():
    assert not hasattr(um, "unsure_question")
    assert not hasattr(um, "resolve_unsure")
    assert not hasattr(um, "UNSURE_ACTIONS")


def test_the_arbiter_asks_jev_once_and_answers():
    answer = {
        "meaning": _p("other", 0.5, other=0.5, answerable=0.3, unformed=0.2),
        "action_other": _p("reply_light", 0.4, reply_light=0.4, reply_full=0.3, ask_back=0.3),
    }
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answer))
    writer = writer_says({"text": "いいよ、しりとりしよう！"})
    d = asyncio.run(
        decide(
            jev=jev, writer=writer, utterance="パジュ、しりとりしよ", origin="発話", family_md=""
        )
    )
    assert jev.ask.await_count == 1
    assert (d.branch, d.text) == ("light", "いいよ、しりとりしよう！")


def test_a_meaning_that_was_not_offered_goes_to_the_main_llm():
    """カメラが無いのに「見る」が返った：使わずに考えて返す（2 回目の問いは外したので、もう聞き直さない）。"""
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True, answers={"meaning": _p("look", 0.9), "action_look": _p("look", 0.9)}
        )
    )
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"text": "x"}),
            utterance="そっち見て",
            origin="発話",
            can_see=False,
        )
    )
    assert jev.ask.await_count == 1
    assert d.branch == "full"


# --- 意味の側も和で比べる（出-db 段 A・2026-10-25・本人の決定ア：見直しの指摘「弱く選ばれた意味で道具が動く」）---


def test_a_weak_tool_meaning_gives_way_to_the_answering_meanings():
    """話しかけ方 0.30 を選んでも、その他 0.28 ＋ 答えられる 0.25 がまされば、返す側の意味で決める（30 分黙らない）。"""
    got = um.settle_meaning(
        _p("speaking", 0.3, speaking=0.3, other=0.28, answerable=0.25, music=0.17)
    )
    assert got["choice"] == "other"


def test_a_clear_tool_meaning_stays():
    """**反証の側**：道具の意味が返す側の和よりまされば、そのまま。"""
    got = um.settle_meaning(_p("music", 0.7, music=0.7, other=0.2, answerable=0.1))
    assert got["choice"] == "music"


def test_silence_names_and_unread_words_are_not_overridden():
    """**反証の側**：成立しない・文脈に合わない・名乗り・否定は、返す側の和では覆さない（道具の意味ではない）。"""
    for m in ("unformed", "off_context", "claim", "deny"):
        got = um.settle_meaning(_p(m, 0.4, **{m: 0.4, "other": 0.3, "answerable": 0.3}))
        assert got["choice"] == m


def test_the_arbiter_uses_the_settled_meaning():
    answer = {
        "meaning": _p("speaking", 0.3, speaking=0.3, other=0.28, answerable=0.25, music=0.17),
        "action_speaking": _p("quiet", 0.6, quiet=0.6, ask_back=0.4),
        "action_other": _p("reply_light", 0.7, reply_light=0.7, ask_back=0.3),
    }
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answer))
    d = asyncio.run(
        decide(jev=jev, writer=writer_says({"text": "うん"}), utterance="あのね", origin="発話")
    )
    assert (d.branch, d.text, d.silence_minutes) == ("light", "うん", 0)  # 0 は黙らない（既定）
