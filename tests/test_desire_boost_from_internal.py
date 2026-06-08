"""Tests for boost-from-internal-result logic."""

from __future__ import annotations

from familiar_agent.agent import EmbodiedAgent


def _boost(text: str) -> float:
    return EmbodiedAgent._boost_from_internal_result(text)


def test_no_notable_content_returns_zero():
    assert _boost("平和な部屋です。") == 0.0
    assert _boost("Nothing special here.") == 0.0
    assert _boost("") == 0.0


def test_single_keyword_returns_small_boost():
    result = _boost("部屋に新しい植物がある。")
    assert result == 0.15


def test_two_keywords_return_medium_boost():
    result = _boost("面白いものを見つけました。")
    assert result == 0.25


def test_three_or_more_keywords_return_large_boost():
    result = _boost("面白い発見をしました。気になることが見つかりました。")
    assert result == 0.35


def test_english_keywords_trigger_boost():
    result = _boost("I found something interesting here.")
    assert result > 0.0


def test_mixed_keywords_accumulate():
    result = _boost("Something interesting was discovered and it's curious.")
    assert result >= 0.25


def test_boost_capped_at_max():
    # Many keywords should not exceed 0.35
    text = " ".join(["面白い", "発見", "気づい", "見つけ", "変化", "新しい"] * 3)
    assert _boost(text) == 0.35


def test_case_insensitive_english():
    assert _boost("I FOUND something.") > 0.0
    assert _boost("Very INTERESTING.") > 0.0
