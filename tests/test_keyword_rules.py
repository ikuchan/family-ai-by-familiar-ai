"""語の列（記-k・2026-09-21）。純関数。

想起は埋め込みの近さだけで候補を集めていた。埋め込みは文全体の似かたを見るので、問いの形
（「覚えてる？」）に引かれ、**中身の語**（本・キャンプ）では引けない。実機 15:58：5 分前に
自分が話した「…キャンプの準備…本を更新…」が **822 位**（類似 0.235）で候補 50 件にも入らず、
パジュは「思い出せない」と答えた。7/25 の「昨日の天気覚えてる？」のほうが上位だった。

そこで**語の軸**を候補の軸として足し、軸ごとの順位で**底上げ**する（`1/(k+順位)` を足す）。
"""

from __future__ import annotations

from familiar_agent.core.keyword_rules import drop_common, pick_words, rank_boost


# ── 問いから語を取り出す ───────────────────────────────────────────────────


def test_only_content_nouns_are_taken():
    # 「さっき」は落ちる（細分が『副詞可能』）。「話し」はこの文では動詞なので取れない。
    assert pick_words("さっき本の話したよね？覚えてる？") == ["本"]


def test_time_words_are_not_taken():
    """「さっき」「前」「いま」は辞書では名詞だが細分が『副詞可能』——時を指すだけ。

    中身のある名詞（「話」）は取れる。ありふれ具合は次の段（`drop_common`）で見る。
    """
    for q, word in (("さっきの話", "さっき"), ("この前の話", "前"), ("いまの話", "いま")):
        got = pick_words(q)
        assert word not in got, (q, got)
        assert "話" in got, (q, got)


def test_several_nouns_come_back_in_order():
    assert pick_words("去年の夏のキャンプの写真どこ？") == ["夏", "キャンプ", "写真"]


def test_nothing_from_a_question_without_nouns():
    assert pick_words("そうなんだ") == []
    assert pick_words("") == []


def test_the_same_word_is_not_repeated():
    got = pick_words("本の話、本当にその本？")
    assert got.count("本") == 1, got


# ── ありふれた語は使わない ─────────────────────────────────────────────────


def test_a_word_that_hits_too_much_is_dropped():
    """この機の実測：本 1,423 件（12.4%）・前 2,389 件（20.9%）・キャンプ 60 件（0.5%）。"""
    counts = {"本": 1423, "前": 2389, "キャンプ": 60}
    words = ["本", "前", "キャンプ"]
    assert drop_common(words, counts, total=11433, max_ratio=0.25) == ["本", "前", "キャンプ"]
    assert drop_common(words, counts, total=11433, max_ratio=0.15) == ["本", "キャンプ"]
    assert drop_common(words, counts, total=11433, max_ratio=0.10) == ["キャンプ"]


def test_a_word_nobody_ever_said_is_dropped():
    assert drop_common(["散歩"], {"散歩": 0}, total=11433, max_ratio=0.25) == []


# ── 軸ごとの順位で底上げする ───────────────────────────────────────────────


def test_nothing_gets_no_lift():
    """どの軸にも載っていなければ底上げはゼロ（いまの採点のまま）。"""
    assert rank_boost([]) == 0.0


def test_the_first_place_of_one_axis():
    """1 つの軸で 1 位なら `1/(10+1)`。"""
    assert rank_boost([1]) == 1.0 / 11.0


def test_two_axes_lift_more_than_one():
    """重なるものは底上げがきつくなる（同じ 1 位でも 2 軸なら 2 倍）。"""
    assert rank_boost([1, 1]) == 2.0 * rank_boost([1])


def test_a_high_place_lifts_more_than_a_low_one():
    """同じ軸でも、上位のほうが強く持ち上がる。"""
    assert rank_boost([1]) > rank_boost([50])


def test_the_lift_of_a_low_place_is_small():
    """50 位の底上げは 7 件の境目（実測 0.054）を動かさない程度に小さい。"""
    assert rank_boost([50]) < 0.02


def test_k_can_be_given():
    """定数は渡せる（k が大きいほど順位の差が縮む）。"""
    assert rank_boost([1], k=60) == 1.0 / 61.0
    assert rank_boost([1], k=60) - rank_boost([50], k=60) < rank_boost([1]) - rank_boost([50])
