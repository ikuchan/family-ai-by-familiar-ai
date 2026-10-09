"""世の中のことでも、作業状態に十分新しい結果があるならそれで答える（2026-09-13 実機で露見）。

「明日の天気は？」を 1 時間後にもう一度聞いたら、11:24 の返事「明日 9 月 14 日の東京は
晴れ・最高 30℃…」が記憶に生きているのに、調停は 1 回目で検索へ行った。プロンプト末尾の
「世の中のこと（天気・ニュース・調べもの）は search_deferred」が無条件だったため。
"""

from __future__ import annotations

from familiar_agent.core import utterance_meaning as um

# 段 5e（2026-10-10）で古い分岐の目安（`JUDGE_GUIDE`）を外した。同じ決まりは、いまは発話の意味の説明にある：
# 「調べる必要のある問い」は答えをまだ持っていないか、変わることで前の答えが古いとき、「知っていることで答えられる問い」は
# 変わることでもその答えがまだ新しいとき。


def test_the_world_rule_is_conditional_on_fresh_material() -> None:
    research = um.MEANINGS["research"][1]
    answerable = um.MEANINGS["answerable"][1]
    assert "まだ持っていない" in research and "古い" in research
    assert "まだ新しい" in answerable


def test_searching_is_only_for_what_is_not_known_or_stale() -> None:
    assert "前の答えが古いとき" in um.ACTIONS["search_deferred"]
