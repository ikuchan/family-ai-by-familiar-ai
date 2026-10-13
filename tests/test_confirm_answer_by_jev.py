"""確認待ちの答えは Jev の 3 択で読む（知-o・2026-10-13・本人の決定）。

10/13 20:00:17 に、確認待ちへの「はい」を発話の意味の大きな問い（10 個以上の選択肢）の 1 つとして読み、確信度が足りずに
聞き返した。確認待ちのあいだは「はい・いいえ・それ以外」の 3 択だけを聞く。倒し先は「それ以外」（掛けもやめもせず促すだけ）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges

QUESTION = "3 分のタイマーね。その間は黙って聞かないよ、いい？"


def _client(pick: "str | None" = None, conf: float = 0.8, *, available: bool = True):
    c = MagicMock()
    c.available = available
    c.ask = AsyncMock(
        return_value=JevAnswer(ok=True, answers={"answer": {"choice": pick, "confidence": conf}})
    )
    return c


def _judge(client, text: str = "はい") -> str:
    return asyncio.run(
        jev_judges.judge_confirm_answer(client, question=QUESTION, text=text, min_conf=0.6)
    )


@pytest.mark.parametrize(
    "pick, verdict",
    [
        ("yes", jev_judges.CONFIRM_YES),
        ("no", jev_judges.CONFIRM_NO),
        ("other", jev_judges.CONFIRM_OTHER),
    ],
)
def test_the_three_choices(pick, verdict):
    assert _judge(_client(pick)) == verdict


def test_the_question_and_the_words_are_sent():
    client = _client("yes")
    _judge(client, "うん、お願い")
    state = client.ask.await_args.args[0]
    assert QUESTION in state and "うん、お願い" in state


def test_a_low_confidence_falls_to_other():
    """**反証の側**：確信度がしきい値未満なら、掛けもやめもしない。"""
    assert _judge(_client("yes", conf=0.59)) == jev_judges.CONFIRM_OTHER


def test_no_jev_falls_to_other():
    assert _judge(_client("yes", available=False)) == jev_judges.CONFIRM_OTHER
    assert _judge(None) == jev_judges.CONFIRM_OTHER
