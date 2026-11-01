"""人の言葉への軽い返事は、これから調べる・やると約束しない（出-dg は・2026-11-01・本人の決定）。

10/31 21:52、主LLM の「探し直しましょうか」にパパが「はい」と答え、状態を伝える返事（`state_light`）が「改めて詳しく
お調べします」「少々お待ちください」と、できない調べものを約束した。聞き返しは出-bk で止めてあったが、状態を伝える・
軽く返すには止める言葉が無かった。黙る依頼・名乗りの返事も軽く返すで書かれ、それらは本当にやることを伝えるので入れない。
"""

from __future__ import annotations

import asyncio

import pytest

from tests._arbiter_fakes import decide, prompt_of, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev

_RULE = "いま進んでいること"


def _write(answers: dict, *, utterance: str = "はい", origin: str = "発話"):
    writer = writer_says({"text": "うん"})
    asyncio.run(decide(jev=_jev(answers), writer=writer, utterance=utterance, origin=origin))
    return prompt_of(writer)


def test_a_status_reply_is_told_not_to_promise():
    prompt = _write(
        {"meaning": _a("accepted_check", 0.9), "action_accepted_check": _a("state_light", 0.9)}
    )
    assert (
        "待ってと約束しない" in prompt and _RULE in prompt
    )  # 当日の「少々お待ちください」も止める


def test_a_light_reply_is_told_not_to_promise():
    prompt = _write({"meaning": _a("other", 0.9), "action_other": _a("reply_light", 0.9)})
    assert "約束しない" in prompt and _RULE in prompt


@pytest.mark.parametrize(
    ("action", "utterance"), [("quiet", "10分黙って"), ("lift_quiet", "もう話していいよ")]
)
def test_accepting_a_quiet_request_is_not_held_back(action, utterance):
    """**反証の側**：黙る依頼・解く言葉は本当にそうするので、「しばらく黙ってるね」と言えるままにする。"""
    prompt = _write(
        {
            "meaning": _a("speaking", 0.9),
            "action_speaking": _a(action, 0.9),
            "quiet_minutes": _a("10", 0.8),
        },
        utterance=utterance,
    )
    assert (
        "次にすることはもう決まっている：軽く返す" in prompt
    )  # 書き手まで届いた（調停が軽く返すに置き換えた）
    assert _RULE not in prompt


def test_asking_back_keeps_its_own_words():
    """**反証の側**：聞き返しは、いまの問いの文（出-bk）のまま。"""
    prompt = _write({"meaning": _a("other", 0.9), "action_other": _a("ask_back", 0.9)})
    assert _RULE not in prompt
