"""聞き返すかどうかを、足切りして割り直した確率で決める（環-at・2026-10-26・本人の決定）。

選択肢が増えると、Jev の確率は 3 つ、4 つに散り、どれもしきい値を越えられずに聞き返しへ倒れていた。0.1〔仮〕未満の
選択肢を 0 にし、残りを足して 1 になるように割り直してから決める。

- 意味：選んだ意味の確率（返す側の意味なら和）が 0.3〔仮〕に届かなければ聞き返す。成立しない・文脈に合わないは黙る。
- 動作：聞き返すは 0.7〔仮〕以上のときだけ採り、届かなければ 0 にして割り直す。決まった動作（返す側なら和）が 0.3 に
  届かなければ聞き返す。
- 名乗り・否定：誰かの答えを割り直し、いちばん上が 0.3 に届かなければ聞き返す（確信度 0.6 をやめた）。

確認の答え・聞き取りの確認の答えは、ほかの判定と同じ設定の下限（`jev_confidence_min`）を見る（一本化）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core import jev_judges
from familiar_agent.core import utterance_meaning as um
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev
from tests.test_event_loop import _agent


def _p(choice: str, **probs: float) -> dict:
    return {"choice": choice, "confidence": 0.5, "probabilities": probs}


# --- 1. 足切りと割り直し ---


def test_trim_cuts_the_thin_options_and_renormalizes():
    got = um.trim(_p("a", a=0.5, b=0.45, c=0.05))
    assert set(got["probabilities"]) == {"a", "b"}
    assert got["probabilities"]["a"] == pytest.approx(0.5 / 0.95)
    assert (
        got["choice"] == "a" and got["confidence"] == 0.5
    )  # 確信度は書き換えない（「たぶん」の言い方は変わらない）


def test_trim_leaves_answers_without_thin_options_or_probabilities():
    """**反証の側**：0.1 未満が無ければ変わらない。確率が付いていない答えも変わらない。"""
    whole = _p("a", a=0.6, b=0.4)
    assert um.trim(whole)["probabilities"] == {"a": 0.6, "b": 0.4}
    assert um.trim(_a("a", 0.2)) == _a("a", 0.2)


# --- 2. 聞き返すは 0.7 以上だけ採る ---


def test_a_weak_ask_back_is_not_taken():
    got = um.decide(_a("other", 0.9), _p("ask_back", ask_back=0.6, reply_light=0.4))
    assert got.final == "reply_light"


def test_a_strong_ask_back_is_taken():
    """**反証の側**：聞き返すが 0.75 なら聞き返す。"""
    got = um.decide(_a("other", 0.9), _p("ask_back", ask_back=0.75, reply_light=0.25))
    assert got.final == "ask_back"


# --- 3. 決まった動作が 0.3 に届かなければ聞き返す ---


def test_an_action_split_four_ways_asks_back():
    split = _p("play_music", play_music=0.25, stop_music=0.25, next_track=0.25, music_volume=0.25)
    assert um.decide(_a("music", 0.9), split).final == "ask_back"


def test_a_leading_action_is_taken():
    """**反証の側**：いちばん上が 0.35 ならその動作。"""
    lead = _p("play_music", play_music=0.35, stop_music=0.25, next_track=0.2, music_volume=0.2)
    assert um.decide(_a("music", 0.9), lead).final == "play_music"


# --- 4. 意味が 0.3 に届かなければ聞き返す ---


def test_a_split_meaning_is_too_weak():
    assert um.too_weak(_p("time", time=0.25, research=0.25, look=0.25, speaking=0.25))
    assert not um.too_weak(
        _p("other", other=0.2, answerable=0.2, time=0.3, research=0.3)
    )  # 返す側は和


def test_silence_meanings_are_never_too_weak():
    """**反証の側**：成立しない・文脈に合わないは、確率が低くても黙る道へ行く（線と比べない）。"""
    assert not um.too_weak(_p("unformed", unformed=0.25, time=0.25, research=0.25, look=0.25))
    assert not um.too_weak(_p("off_context", off_context=0.25, time=0.25, look=0.25, other=0.25))


# --- 5. 名乗り・否定は、誰かの答えを割り直して 0.3 を見る ---


def test_a_claim_with_a_leading_name_is_taken():
    names = _p("パパ", パパ=0.47, ママ=0.39, other=0.13)
    got = um.decide(_a("claim", 0.5), names)
    assert (got.final, got.name) == ("claim", "パパ")


def test_a_claim_split_four_ways_asks_back():
    """**反証の側**：誰かが 4 人に 0.25 ずつ割れたら聞き返す。"""
    names = _p("パパ", パパ=0.25, ママ=0.25, おばあちゃん=0.25, other=0.25)
    assert um.decide(_a("claim", 0.9), names).final == "ask_back"


# --- 6. 調停を通す ---


def _timer(monkeypatch, meaning: dict, action: dict):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"meaning": meaning, "action_time": action})
    writer = writer_says({"text": "3分のタイマーのこと？", "tool_input": {"after_minutes": 3}})
    return asyncio.run(
        decide(jev=jev, writer=writer, utterance="3分測って", extra_actions=("set_timer",))
    )


def test_the_arbiter_does_not_ask_back_on_thin_probabilities(monkeypatch):
    meaning = _p("time", time=0.6, other=0.35, unformed=0.05)
    action = _p("ask_back", ask_back=0.5, set_timer=0.45, silent=0.05)
    d = _timer(monkeypatch, meaning, action)
    assert (d.branch, d.action) == ("action", "set_timer")


def test_the_arbiter_asks_back_on_a_split_meaning(monkeypatch):
    """**反証の側**：意味が 4 つに割れたら、動作が強くても聞き返す。"""
    meaning = _p("time", time=0.25, research=0.25, look=0.25, speaking=0.25)
    d = _timer(monkeypatch, meaning, _p("set_timer", set_timer=0.95, silent=0.05))
    assert (d.branch, d.text) == ("light", "3分のタイマーのこと？")


# --- 7〜9. 確認の答えは設定の下限（jev_confidence_min）を見る ---


def _floors(monkeypatch) -> list:
    """確認の答えの判定に渡った下限を残す（倒れた理由が下限であることを確かめる）。"""
    seen: list = []
    real = jev_judges.judge_confirm_answer

    async def spy(*a, **kw):
        seen.append(kw["min_conf"])
        return await real(*a, **kw)

    monkeypatch.setattr(jev_judges, "judge_confirm_answer", spy)
    return seen


def _confirm(monkeypatch, conf: float, **kw):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    arbiter = Arbiter(jev=_jev({"answer": _a("yes", conf)}), writer=writer_says(), **kw)
    inp = ArbiterInput(
        utterance="うん",
        workspace_ctx="",
        confirm_question="3 分のタイマーね。いい？",
        confirm_prompt="いい？",
        confirm_prompt_due=False,
    )
    return asyncio.run(arbiter.decide(inp))


def test_the_confirm_answer_uses_the_configured_floor(monkeypatch):
    floors = _floors(monkeypatch)
    d = _confirm(monkeypatch, 0.7, min_conf=0.8)
    assert floors == [0.8]
    assert (d.branch, d.text) == ("light", "")  # 「はい」と読まず、促す間隔が空いていないので黙る


def test_the_confirm_answer_passes_above_the_default_floor(monkeypatch):
    """**反証の側**：既定（0.6）なら、同じ確信度 0.7 の「はい」で掛ける。"""
    floors = _floors(monkeypatch)
    assert _confirm(monkeypatch, 0.7).action == "confirm"
    assert floors == [0.6]


_HEARD = {
    "heard_check_action": "set_timer",
    "heard_check_utterance": "3分測って",
    "heard_check_question": "3分のタイマーのこと？",
}


def _heard(monkeypatch, conf: float, **kw):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    writer = writer_says({"tool_input": {"after_minutes": 3, "label": ""}})
    arbiter = Arbiter(jev=_jev({"answer": _a("yes", conf)}), writer=writer, **kw)
    inp = ArbiterInput(utterance="うん", workspace_ctx="", extra_actions=("set_timer",), **_HEARD)
    return asyncio.run(arbiter.decide(inp))


def test_the_heard_check_answer_uses_the_configured_floor(monkeypatch):
    floors = _floors(monkeypatch)
    d = _heard(monkeypatch, 0.7, min_conf=0.8)
    assert floors == [0.8]
    assert (
        d.action != "set_timer"
    )  # 「はい」と読まず、ふつうの道へ回る（偽の Jev に意味の答えが無いので倒し先になる）


def test_the_heard_check_answer_passes_above_the_default_floor(monkeypatch):
    """**反証の側**：既定（0.6）なら、同じ「はい」で見立ての動作を投げる。"""
    floors = _floors(monkeypatch)
    assert _heard(monkeypatch, 0.7).action == "set_timer"
    assert floors == [0.6]


@pytest.mark.parametrize("floor", [0.6, 0.8])
def test_the_arbiter_gets_the_configured_floor(floor):
    from familiar_agent.loop.event_loop import InformationProcessing

    a = _agent(stream_returns=[])
    a.config.jev_confidence_min = floor
    assert InformationProcessing(a)._arbiter()._min_conf == floor


# --- 見直しの指摘（2 回目）---


def test_the_guess_wording_follows_jev_raw_probability(monkeypatch):
    """意味を返す側に置き換えても、「たぶん」は Jev の生の確率（0.55）で決める。割り直した 0.65 で言い切らない。"""
    from tests._arbiter_fakes import prompt_of

    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    meaning = _p("time", time=0.2, other=0.55, answerable=0.1, look=0.08, research=0.07)
    jev = _jev({"meaning": meaning, "action_other": _p("ask_back", ask_back=0.9, reply_light=0.1)})
    writer = writer_says({"text": "なんのこと？"})
    asyncio.run(decide(jev=jev, writer=writer, utterance="あれ", extra_actions=("set_timer",)))
    assert "もしかすると" in prompt_of(writer) and "たぶん" not in prompt_of(writer)
