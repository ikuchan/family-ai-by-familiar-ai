"""情動で話しかけるときは、独り言ではなく話しかけとして、まず話してよいかを尋ねる（出-at・2026-09-30）。

声にする情動は bond・esteem だけ（出-as §2.1）。ところが長さの行は軸に関わらず「[独り言]…誰にも向けない」で、
bond の内声「居れば声をかける」と食い違っていた。話しかけ方（驚かせない・丁寧に・短く・許可を尋ねる）の
指示もどこにも無かった。主LLM の長さの行と、調停が light で閉じるときの軽量LLM の書き方の両方に渡す
（片方だけだと、調停が light を選んだときに長いまま話しかける）。seeking・safety・rest は独り言のまま。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from familiar_agent.core.drive_autonomy import TALKING_AXES
from familiar_agent.loop import reply_budget
from familiar_agent.loop.event_loop import InformationProcessing
from tests._arbiter_fakes import decide, jev_says, prompt_of, writer_says
from tests.test_event_loop import _agent

ASK_FIRST = "まず話してよいかを尋ねる一言"


def test_only_bond_and_esteem_talk():
    assert TALKING_AXES == frozenset({"bond", "esteem"})


def test_a_talking_drive_gets_a_talking_line():
    b = reply_budget.decide(effort="low", researched=False, w_count=3, origin="情動", talking=True)
    assert b.line().startswith("[話しかけ]"), b.line()
    assert "目標 20 字・40 字以内" in b.line()
    assert "驚かない" in b.line() and ASK_FIRST in b.line()
    assert "誰にも向けない" not in b.line()


def test_other_drives_stay_a_soliloquy():
    b = reply_budget.decide(effort="low", researched=False, w_count=3, origin="情動")
    assert b.line().startswith("[独り言]") and "誰にも向けない" in b.line()


def _budget_talking_for(axis: str) -> bool:
    a = _agent(stream_returns=[])
    a._jev = jev_says("full")

    async def scenario():
        ip = InformationProcessing(a)
        ip._req.trigger_kind = "情動"
        ip._req.fired_axis = axis
        ip._req.cue = "誰かと居たい気持ちが湧いている。"
        with patch.object(reply_budget, "decide", wraps=reply_budget.decide) as spy:
            await ip._iterate()
        await ip.close()
        return spy.call_args.kwargs.get("talking", False)

    return asyncio.run(scenario())


def test_the_iteration_tells_the_budget_when_the_drive_talks():
    assert _budget_talking_for("bond") is True
    assert _budget_talking_for("esteem") is True
    assert _budget_talking_for("seeking") is False


def _light_prompt(*, talking: bool) -> str:
    writer = writer_says({"text": "いま、少しいい？"})
    asyncio.run(
        decide(
            jev=jev_says("light"),
            writer=writer,
            utterance="誰かと居たい気持ちが湧いている。",
            origin="情動",
            talking=talking,
        )
    )
    return prompt_of(writer)


def test_the_arbiter_writes_a_talk_when_the_drive_talks():
    prompt = _light_prompt(talking=True)
    assert ASK_FIRST in prompt and "驚かない" in prompt


def test_the_arbiter_keeps_the_self_line_otherwise():
    prompt = _light_prompt(talking=False)
    assert ASK_FIRST not in prompt
    assert "自分から言うなら短いひとこと" in prompt
