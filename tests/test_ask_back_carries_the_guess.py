"""聞き返しには Jev の見立てを渡す（出-bl・2026-10-11・本人の決定ア）。

聞き返しは「Jev が迷った」ときにも倒れる先なのに、書く側は何を迷ったのかを知らず「何をしてほしいのか」という空の問いしか
書けなかった（10/11 08:24「パジュ、今日の予定は?」）。いちばん高い意味と、その意味でいちばん高い動作（聞き返す・黙るを除く）を
言葉にして渡し、見立てが合っているかを確かめる問いにする。数字は渡さない。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core import utterance_meaning as um
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, prompt_of, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev


def _with_probs(choice: str, confidence: float, probs: dict) -> dict:
    return {"choice": choice, "confidence": confidence, "probabilities": probs}


# --- 見立ての作り方 ---


def test_the_guess_names_the_meaning_and_the_top_action_besides_asking():
    meaning = _a("research", 0.55)
    action = _with_probs(
        "ask_back", 0.4, {"ask_back": 0.4, "recall": 0.35, "search_deferred": 0.25}
    )
    got = um.guess(meaning, action)
    assert um.MEANINGS["research"][0] in got
    assert um.ACTIONS["recall"] in got
    assert um.ACTIONS["search_deferred"] not in got
    assert "もしかすると" in got


def test_a_sure_meaning_is_said_as_probably():
    got = um.guess(_a("music", 0.9), _a("play_music", 0.5))
    assert "たぶん" in got and "もしかすると" not in got
    assert um.ACTIONS["play_music"] in got  # 確率が無ければ choice を使う


def test_only_asking_or_silence_leaves_the_action_out():
    got = um.guess(_a("music", 0.9), _with_probs("ask_back", 0.9, {"ask_back": 0.9, "silent": 0.1}))
    assert um.MEANINGS["music"][0] in got
    assert um.ACTIONS["ask_back"] not in got and um.ACTIONS["silent"] not in got


def test_no_guess_for_words_that_do_not_fit_or_are_not_formed():
    assert um.guess(_a("off_context", 0.4), _a("ask_back", 0.9)) == ""
    assert um.guess(_a("unformed", 0.4), None) == ""


def test_a_claim_gives_only_the_meaning():
    got = um.guess(_a("claim", 0.5), _a("パパ", 0.5))
    assert got and um.MEANINGS["claim"][0] in got and "パパ" not in got


# --- 調停が軽量LLM に渡す ---


def _ask_back_after_doubt(monkeypatch, answers: dict):
    kept: dict = {}
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: kept.update(kw))
    jev = _jev(answers)
    writer = writer_says({"text": "家族の予定を見ればいいですか？"})
    d = asyncio.run(
        decide(jev=jev, writer=writer, utterance="パジュ、今日の予定は?", origin="発話")
    )
    return d, prompt_of(writer), kept


def test_a_doubtful_ask_back_tells_the_writer_the_guess(monkeypatch):
    d, prompt, kept = _ask_back_after_doubt(
        monkeypatch,
        {
            "meaning": _a("research", 0.5),
            # 聞き返すが確率でまさる（出-db 段 A）。見立ては聞き返す・黙るを除いた最大の recall。
            "action_research": _with_probs(
                "ask_back", 0.5, {"ask_back": 0.5, "recall": 0.3, "search_deferred": 0.2}
            ),
        },
    )
    assert (d.branch, d.text) == ("light", "家族の予定を見ればいいですか？")
    assert "見立て" in prompt and um.MEANINGS["research"][0] in prompt
    assert um.ACTIONS["recall"] in prompt
    # 聞き取りの確認の形で聞く（出-cm・見立てが合っているかを聞く形から替えた）。
    assert "と言ったの？" in prompt and "約束しない" in prompt
    assert "見立て" in kept["outcome"]


def test_an_off_context_word_is_not_asked_back(monkeypatch):
    """文脈に合わない言葉は、見立ての無い聞き返しをしていた。出-db 段 B から、名前が無ければ黙る（書く側を呼ばない）。"""
    kept: dict = {}
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: kept.update(kw))
    jev = _jev({"meaning": _a("off_context", 0.4), "action_off_context": _a("ask_back", 0.9)})
    writer = writer_says({"text": "なんて？"})
    d = asyncio.run(decide(jev=jev, writer=writer, utterance="あれ", origin="発話"))
    assert (d.branch, d.text) == ("light", "")
    writer.complete.assert_not_awaited()
    assert "黙る" in kept["outcome"]
