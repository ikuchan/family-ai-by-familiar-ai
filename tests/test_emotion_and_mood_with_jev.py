"""感情の評価と気分の見立てを Jev が決める（出-au 段 5-6・2026-09-27・`設計方針_判定の段` §2.2.3）。

- **感情の評価**：Score を 3 問（P 快・Pn 不快・Dom 掌握）、各 5 段で 1 回に聞く。返る値（段の番号を確率で重みづけた
  0〜4）を 0〜1 に直す。平静の位置といまの気分は事実として送る（答えの側を指示しない・出-e）。高ぶり（A）が
  しきい値未満なら聞かない。Jev が使えない・失敗・3 つそろわなければ「未測定」（気分で埋めない・050）。
  確信度では倒さない（Score は迷いが真ん中寄りの値に表れる）。
- **気分の見立て**：Choice（engaged／tired／frustrated／absent／happy）。確信度では倒さない。Jev が使えないときは
  見立てずに既定の `absent`（本人の決定 2026-09-27）。
- 軽量LLM の口（指示文 2 つ・`_evaluate_emotion_pad`）と、語で見る機械の判定は外した。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.mood_register import MoodPAD


def _client(answers=None, *, ok=True):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers or {}))
    return c


_MOOD = MoodPAD(p=0.1, pn=0.1, a=0.5, dom=0.5)


def _emotion(client, arousal=0.6):
    return asyncio.run(
        jev_judges.judge_emotion(
            client, text="パパ：ありがとう！", mood=_MOOD, arousal=arousal, a_gate=0.25
        )
    )


def test_three_scores_become_the_pad():
    c = _client(
        {
            "p": {"type": "score", "score": 3.0},
            "pn": {"type": "score", "score": 0.0},
            "dom": {"type": "score", "score": 2.0},
        }
    )
    pad, a = _emotion(c)
    assert (pad.p, pad.pn, pad.dom, pad.a) == pytest.approx((0.75, 0.0, 0.5, 0.6))
    state, questions = c.ask.await_args.args
    assert set(questions) == {"p", "pn", "dom"}
    assert all(q["type"] == "score" and len(q["criteria"]) == 5 for q in questions.values())
    assert "ありがとう" in state and "0.10" in state  # いまの気分（事実）を送る


def test_low_arousal_is_not_asked():
    c = _client()
    pad, a = _emotion(c, arousal=0.1)
    assert pad is None and a == pytest.approx(0.1)
    c.ask.assert_not_awaited()


def test_failure_or_missing_scores_are_unmeasured():
    assert _emotion(_client(ok=False))[0] is None
    assert _emotion(_client({"p": {"score": 1.0}}))[0] is None
    assert _emotion(None)[0] is None


def _mood(client, text="今日はもう疲れたよ"):
    return asyncio.run(jev_judges.judge_companion_mood(client, text=text))


def test_the_companion_mood_is_chosen_without_a_confidence_cut():
    assert _mood(_client({"mood": {"choice": "tired", "confidence": 0.3}})) == "tired"


def test_without_jev_the_default_is_absent():
    assert _mood(None) == "absent"
    assert _mood(_client(ok=False)) == "absent"
    assert _mood(_client({"mood": {"choice": "angry", "confidence": 0.9}})) == "absent"


def test_the_evaluator_uses_jev_and_the_light_llm_parts_are_gone():
    from familiar_agent.loop import evaluator

    ev = evaluator.Evaluator(MagicMock(), MagicMock(), jev=_client({"mood": {"choice": "happy"}}))
    assert asyncio.run(ev.infer_companion_mood("やったー！")) == "happy"
    for gone in (
        "_evaluate_emotion_pad",
        "_EMOTION_PAD_PROMPT",
        "_COMPANION_MOOD_PROMPT",
        "_companion_mood_heuristic",
    ):
        assert not hasattr(evaluator, gone), gone
