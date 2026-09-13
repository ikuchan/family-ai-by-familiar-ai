"""世の中のことでも、作業状態に十分新しい結果があるならそれで答える（2026-09-13 実機で露見）。

「明日の天気は？」を 1 時間後にもう一度聞いたら、11:24 の返事「明日 9 月 14 日の東京は
晴れ・最高 30℃…」が記憶に生きているのに、調停は 1 回目で検索へ行った。プロンプト末尾の
「世の中のこと（天気・ニュース・調べもの）は search_deferred」が無条件だったため。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT


def test_the_world_rule_is_conditional_on_fresh_material() -> None:
    at = ARBITER_PROMPT.index("世の中のこと")
    rule = ARBITER_PROMPT[at : at + 260]
    assert "十分新しい" in rule or "新しい" in rule
    assert "それで答える" in rule
    assert "無い" in rule and "古い" in rule


def test_the_old_unconditional_wording_is_gone() -> None:
    assert (
        '世の中のこと（天気・ニュース・調べもの）は "search_deferred" を選ぶ。'
        not in ARBITER_PROMPT
    )
