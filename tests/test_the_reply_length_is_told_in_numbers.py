"""返事の長さは規則の文言でなく、数字で渡す（出-k-ろ）。"""

from __future__ import annotations

from familiar_agent.loop.generator import _iter_ctx
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT
from familiar_agent.loop.reply_budget import ReplyBudget


def test_the_iteration_line_carries_the_budget() -> None:
    text = _iter_ctx(
        chain=1,
        max_chain=5,
        thinking_round=1,
        capped=False,
        budget=ReplyBudget(target=40, limit=80, max_tokens=500),
    )
    assert "[返事] 目標 40 字・80 字以内" in text


def test_the_old_wording_is_gone() -> None:
    assert "1〜2文" not in EVENT_SYSTEM_PROMPT
    assert "[返事]" in EVENT_SYSTEM_PROMPT, "長さは [返事] の行に従う、と案内する"
