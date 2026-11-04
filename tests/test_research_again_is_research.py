"""出-dl 原因 1：「もう一度調べてみて」を、既に受けた作業依頼の確認ではなく、調べる必要のある問いに読ませる。

2026-11-03 19:12、「もう一度調べてみて」が確認（accepted_check）0.69・調べる（research）0.30 に読まれ、軽い返事が
「確かめます・お待ちください」と約束して黙った。確認の説明にあった「続けてほしいか」を外し、調べ直しの頼みは
調べる側に入れる。11/04 の試し（記録 36 件×3 回・jev-1.14.0）で、「もう一度調べてみて」「調べて」は調べるに寄り、
「さっきの調べもの、どうなった？」は確認のまま残った（本人の決定ア）。
"""

from __future__ import annotations

from familiar_agent.core import utterance_meaning as um


def test_the_check_meaning_leaves_out_asking_again():
    text = um.MEANINGS["accepted_check"][1]
    assert "調べ直し・もう一度調べての頼みは含まない" in text
    assert "続けてほしいか" not in text


def test_the_check_meaning_still_checks_what_was_done():
    """**反証の側**：確認は、頼んだことがどうなったか・やってくれたかを確かめる言葉のまま。"""
    text = um.MEANINGS["accepted_check"][1]
    assert "どうなったか" in text and "念を押す" in text


def test_research_includes_asking_to_search_again():
    text = um.MEANINGS["research"][1]
    assert "前に調べたことを調べ直して・もう一度調べてと頼むのも含む" in text
    assert "前の答えが古い" in text
