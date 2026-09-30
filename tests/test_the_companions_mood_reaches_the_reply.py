"""相手の気分の見立てを、主LLM の system 文へ一行で渡す（出-av・2026-09-30）。

見立ての判定（`jev_judges.judge_companion_mood`・出-au 段 5-6）はあったが、イベント駆動ループはどこも呼んで
いなかった。会話の求めで主LLM を呼ぶとき、相手の言葉から Jev が見立て、`[相手の様子]` を在席の隣に置く
（本人の決定ア：返事の調子を相手に合わせる）。分からないとき（absent・Jev が使えない）は何も置かない。
情動と機器の求めでは見立てない（相手の言葉が無い）。数字は渡さない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.evaluator import Evaluator
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.generator import _companion_ctx
from tests._arbiter_fakes import jev_says
from tests.test_event_loop import _agent, _run, _turn


@pytest.mark.parametrize(
    "mood,words",
    [
        ("engaged", "乗り気で話している"),
        ("happy", "うれしそう・楽しそう"),
        ("tired", "疲れていそう"),
        ("frustrated", "苛立っているか、困っていそう"),
    ],
)
def test_each_mood_becomes_words(mood, words):
    assert _companion_ctx(mood, "") == f"[相手の様子] 相手は{words}"
    assert _companion_ctx(mood, "パパ") == f"[相手の様子] パパは{words}"


def test_an_unknown_mood_puts_nothing():
    assert _companion_ctx("absent", "パパ") == ""
    assert _companion_ctx("sleepy", "") == ""


def _ip(kind: str, evaluator):
    a = _agent(stream_returns=[])
    a._evaluator = evaluator
    ip = InformationProcessing(a)
    ip._req.trigger_kind = kind
    return ip


def test_a_conversation_asks_for_the_mood():
    ev = MagicMock(infer_companion_mood=AsyncMock(return_value="tired"))
    assert asyncio.run(_ip("発話", ev)._judge_companion("今日は疲れたよ")) == "tired"
    ev.infer_companion_mood.assert_awaited_once_with("今日は疲れたよ")


@pytest.mark.parametrize("kind", ["情動", "機器"])
def test_affect_and_devices_are_not_judged(kind):
    ev = MagicMock(infer_companion_mood=AsyncMock(return_value="tired"))
    assert asyncio.run(_ip(kind, ev)._judge_companion("誰かと居たい")) == "absent"
    ev.infer_companion_mood.assert_not_awaited()


def test_a_failing_judge_puts_nothing():
    ev = MagicMock(infer_companion_mood=AsyncMock(side_effect=RuntimeError("down")))
    assert asyncio.run(_ip("発話", ev)._judge_companion("今日は疲れたよ")) == "absent"


def _system_with_jev_mood(mood: "str | None") -> str:
    from familiar_agent.backends import ToolCall

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "お疲れさま"})])])
    jev = jev_says("full")
    if mood is not None:
        jev.ask.return_value.answers["mood"] = {"choice": mood, "confidence": 0.9}
    a._jev = jev
    a._evaluator = Evaluator(MagicMock(), MagicMock(), jev=jev)
    a._persons.active_is_explicit = False  # 話者は分からない（「相手」と書く）
    _run(a, utterance="今日は疲れたよ")
    return "\n".join(a.backend.stream_turn.call_args.kwargs["system"])


def test_the_main_llm_is_told_how_the_companion_seems():
    assert "[相手の様子] 相手は疲れていそう" in _system_with_jev_mood("tired")


def test_nothing_is_told_when_the_mood_is_unknown():
    assert "[相手の様子]" not in _system_with_jev_mood(None)
