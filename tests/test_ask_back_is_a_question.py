"""聞き返しの文は問いにする。問いでなければ主LLM へ（出-bk・2026-10-11・本人の決定ア）。

10/11 08:24:19「パジュ、今日の予定は?」が聞き返しに倒れ、軽量LLM が「調べますので、少々お待ちください。」と書いた。
聞き返しの道は道具を投げないので、何も調べずに返事が来なかった。書く欄の説明が「短く答える」だった。聞き返しには
問いの説明を渡し、返った文が「？」で終わっていなければ捨てて主LLM に考えて返させる。
"""

from __future__ import annotations

import asyncio

import pytest

from tests._arbiter_fakes import decide, prompt_of, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev


def _ask_back(text: str):
    jev = _jev({"meaning": _a("music", 0.9), "action_music": _a("ask_back", 0.9)})
    writer = writer_says({"text": text})
    d = asyncio.run(
        decide(jev=jev, writer=writer, utterance="パジュ、今日の予定は?", origin="発話")
    )
    return d, writer


def test_a_promise_instead_of_a_question_goes_to_the_main_llm():
    d, _ = _ask_back("調べますので、少々お待ちください。")
    assert d.branch == "full" and not d.text


@pytest.mark.parametrize(
    "text", ["何の予定を知りたいですか？", "どの曲にする?", "今日の、どれのこと？」"]
)
def test_a_question_is_spoken(text):
    d, _ = _ask_back(text)
    assert (d.branch, d.text) == ("light", text)


def test_the_writer_is_told_to_ask_not_to_promise():
    _, writer = _ask_back("何の予定を知りたいですか？")
    prompt = prompt_of(writer)
    assert "問い" in prompt and "約束しない" in prompt
