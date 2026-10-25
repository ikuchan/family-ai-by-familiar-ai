"""文脈に合わない言葉には黙る（出-db 段 B・2026-10-25・本人の決定 3-e・`設計方針_判定の段` §2.2.5）。

以前は、聞き返すを 0.6 以上で選んだときだけ聞き返していた。10/24 はテレビや崩れた声に 7 回聞き返した。名前の無い言葉なら
いつも黙る。名前で呼ばれた言葉は話しかけられているので、次に確率の高い意味で読む。意味の説明は「続きとしても新しい話題と
しても通らない」に、問いの文は「話題を変えた言葉も、その中身で選ぶ」に置き換えた（「ところで」の除外は、実験で
文脈に合わないを 0.54〜0.92 に押し上げて逆効果だった）。
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


def _run(answer: dict, *, named: bool):
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answer))
    writer = writer_says({"text": "あれって何のこと？"})
    d = asyncio.run(decide(jev=jev, writer=writer, utterance="あれ", origin="発話", named=named))
    return d, writer, jev


def test_off_context_is_silent_even_when_asking_back_is_sure():
    assert um.decide(_p("off_context", 0.5), _p("ask_back", 0.9)).final == "silent"


def test_off_context_asks_no_second_question():
    qs = um.fanout_questions(music=False, camera=False, family=["パパ"])
    assert "action_off_context" not in qs


def test_an_unnamed_off_context_word_is_silent():
    d, writer, _ = _run(
        {"meaning": _p("off_context", 0.6, off_context=0.6, other=0.4)}, named=False
    )
    assert (d.branch, d.text) == ("light", "")
    writer.complete.assert_not_awaited()


def test_a_named_off_context_word_is_read_by_the_next_meaning():
    """**反証の側**：名前で呼ばれたら黙らない。次に確率の高い意味（その他の会話）で読む。"""
    answer = {
        "meaning": _p("off_context", 0.5, off_context=0.5, other=0.3, unformed=0.15, music=0.05),
        "action_other": _p("reply_light", 0.8, reply_light=0.8, ask_back=0.2),
    }
    d, _, _ = _run(answer, named=True)
    assert (d.branch, d.text) == ("light", "あれって何のこと？")


def test_the_new_wording():
    desc = um.MEANINGS["off_context"][1]
    assert desc == "直前のやりとりの続きとしても、新しい話題としても、何を言いたいのか通らない"
    qs = um.fanout_questions(music=False, camera=False, family=["パパ"])
    instr = qs["meaning"]["instructions"]
    assert "話題を変えた言葉も、その中身で選ぶ" in instr
    assert "思い出したことは文脈に含めない" in instr
    assert "ところで" not in instr and "ところで" not in desc


# --- 読み直しの足切り（出-db 段 B・2026-10-25・本人の決定ア・線 0.25）---


def test_a_named_request_is_reread_above_the_cut():
    """10/24 18:43・名前つきで、ある試合の結果を調べてと頼んだ言葉：文脈に合わない 0.58・調べる 0.36 → 調べる。"""
    got = um.reread_named(
        _p("off_context", 0.58, off_context=0.58, research=0.36, answerable=0.05, deny=0.01)
    )
    assert got["choice"] == "research"


def test_a_named_garble_below_the_cut_stays_silent():
    """**反証の側**：10/24 19:02「パチュうちゅう…これです」：文脈に合わない 0.50・成立しない 0.29・その他 0.13 → 黙る。"""
    got = um.reread_named(
        _p("off_context", 0.5, off_context=0.5, unformed=0.29, other=0.13, answerable=0.03)
    )
    assert got["choice"] == "off_context"
    assert um.decide(got, None).final == "silent"


def test_names_and_denials_are_not_reread_into():
    """名乗り・否定は読み直し先にしない（読み違えて話者を付け替える害が大きい）。"""
    got = um.reread_named(_p("off_context", 0.5, off_context=0.5, deny=0.3, other=0.2))
    assert got["choice"] == "off_context"


def test_the_cut_is_a_quarter():
    assert um.REREAD_MIN == 0.25


# --- 名前つきとして読むのは、名前で呼ばれた声とキーボード（出-db 段 B・本人の決定ア）---


def _arbiter_input(*, source: str, named: bool):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.source, ip._req.named = source, named
    return ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="")


def test_typed_words_are_read_as_named():
    """キーボードで打った言葉は話しかけなので、名前が無くても読み直す（窓の開け閉めは変えない）。"""
    assert _arbiter_input(source="keyboard", named=False).named


def test_a_named_voice_is_named_and_an_unnamed_voice_is_not():
    assert _arbiter_input(source="voice", named=True).named
    assert not _arbiter_input(source="voice", named=False).named  # **反証の側**
