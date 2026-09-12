"""返事の長さと `max_tokens` は、求めごとに規則で決める（出-k-ろ・課題5 G 章）。

既定 目標 40／上限 80 字、high 80／160、ネット調査 160／240（調査が優先）。`max_tokens` は
出力だけの上限で、返事・`say` の JSON と申告・内部の思考を含む（実測：思考が要る問いで
150 は思考だけで使い切り本文が空）。
"""

from __future__ import annotations

from familiar_agent.loop.reply_budget import ReplyBudget, decide


def test_the_default_is_short() -> None:
    b = decide(effort="low", researched=False, w_count=7)
    assert (b.target, b.limit) == (40, 80)


def test_medium_is_as_short_as_low() -> None:
    assert decide(effort="medium", researched=False, w_count=7).limit == 80


def test_high_allows_twice() -> None:
    b = decide(effort="high", researched=False, w_count=7)
    assert (b.target, b.limit) == (80, 160)


def test_research_wins_over_effort() -> None:
    assert (
        decide(effort="low", researched=True, w_count=7).target,
        decide(effort="high", researched=True, w_count=7).limit,
    ) == (160, 240)


def test_max_tokens_is_limit_verdicts_and_thinking() -> None:
    # 上限字数×2 ＋ 申告分（件数×20＋60）＋ 思考分（low 256／medium 1024／high 2048）
    assert (
        decide(effort="low", researched=False, w_count=11).max_tokens
        == 80 * 2 + (11 * 20 + 60) + 256
    )
    assert decide(effort="medium", researched=False, w_count=0).max_tokens == 160 + 60 + 1024
    assert decide(effort="high", researched=True, w_count=3).max_tokens == 480 + 120 + 2048


def test_an_unknown_effort_is_treated_as_low() -> None:
    assert decide(effort="", researched=False, w_count=0).max_tokens == 160 + 60 + 256


def test_the_budget_reads_as_one_line() -> None:
    assert ReplyBudget(target=40, limit=80, max_tokens=500).line() == "[返事] 目標 40 字・80 字以内"
