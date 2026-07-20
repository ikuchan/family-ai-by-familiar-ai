"""Tests for the arousal scale of AppraisalEngine（感情語の飽和）.

`_count_patterns` は一致語数を辞書サイズで割っていたため、感情語が2つあっても
joy=0.25 にしかならず、arousal は 0.1375 にとどまっていた。値踏みゲート
A_GATE=0.25 を越えられず、評価器（軽量LLM）が起動しないまま観測 PAD が中立で
書かれ続けていた。課題5 v0.24 は「A は平常0.5でも感情は動くため省略は深い鎮静
A<0.25 のみ」としており、ゲートが常時省略へ反転していた。

辞書サイズで割ると、語彙を増やすほど感度が下がるという意図と逆の性質も生む。
一致語数を定数2で飽和させる形に変え、感情語がひとつ出れば値踏みが起こるようにする。
"""

from __future__ import annotations

from familiar_agent.appraisal import AppraisalContext, AppraisalEngine


A_GATE = 0.25  # agent.py の値踏みゲート（課題5 v0.24）


def _arousal(text: str) -> float:
    return AppraisalEngine().appraise(AppraisalContext(user_text=text)).arousal


def test_single_joy_word_crosses_the_appraisal_gate() -> None:
    """感情語ひとつで評価器が起動する（joy=0.5 → arousal≈0.275）。"""
    got = _arousal("今日はほんとに嬉しい")
    assert got > A_GATE, f"感情語1つで A がゲートを越えない: {got}"


def test_single_distress_word_crosses_the_appraisal_gate() -> None:
    """沈んだ側もひとことで起動する（distress=0.5 → arousal≈0.30）。"""
    got = _arousal("今日はしんどい")
    assert got > A_GATE, f"感情語1つで A がゲートを越えない: {got}"


def test_two_words_saturate() -> None:
    """2語で飽和する（それ以上増えても頭打ち）。"""
    two = _arousal("最高に楽しい")
    three = _arousal("やった、最高に楽しい、嬉しい")
    assert two > A_GATE
    assert abs(three - two) < 1e-9, "2語で飽和していない"


def test_calm_text_stays_below_the_gate() -> None:
    """感情語の無い淡々としたターンは従来どおりゲート未満（反証側）。"""
    for text in ("おはよう", "今日も暑いね", "うん"):
        got = _arousal(text)
        assert got < A_GATE, f"淡々とした入力で A が上がっている: {text!r} → {got}"


def test_adding_vocabulary_does_not_dull_detection() -> None:
    """辞書サイズに依存しない（語彙を増やしても感度が落ちない）。

    実装が辞書サイズで割っていると、この不変条件が壊れる。
    """
    from familiar_agent import appraisal

    base = _arousal("今日はほんとに嬉しい")
    original = list(appraisal._JOY_PATTERNS)
    try:
        appraisal._JOY_PATTERNS.extend([r"わーい", r"ばんざい", r"最高潮", r"ごきげん"])
        widened = _arousal("今日はほんとに嬉しい")
    finally:
        appraisal._JOY_PATTERNS[:] = original
    assert abs(widened - base) < 1e-9, "語彙を増やすと感度が下がる"
