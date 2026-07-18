"""Tests for expanded emotion vocabulary (Phase 5 — emotional depth).

文字列 mood 側（_MOOD_INTENSITY・_interoception の mood feels・_update_mood）が
12ラベルを網羅することを確認する。感情ラベルは W2b-2 で PAD から派生する形へ
変わったが、この文字列 mood 機構は移行期も生きている。
"""

from __future__ import annotations

import time

from familiar_agent.agent import EmbodiedAgent, _interoception


# The expected full set of emotion labels
_EXPECTED_EMOTIONS = {
    "happy",
    "sad",
    "curious",
    "excited",
    "moved",
    "surprised",
    "nostalgic",
    "relieved",
    "tender",
    "playful",
    "proud",
    "neutral",
}

_NON_NEUTRAL_EMOTIONS = _EXPECTED_EMOTIONS - {"neutral"}


# 旧 _EMOTION_PROMPT / _infer_emotion（ラベル直出し）のテストは、W2b-2 で評価器が
# PAD を出しラベルを PAD から派生する形へ変わったため撤去した。12ラベルの網羅は
# emotion_pad.LABEL_PAD 側（test_emotion_pad_module）で担保する。ここでは文字列 mood の
# _MOOD_INTENSITY・_interoception・_update_mood（現存機構）の網羅を引き続き確認する。


# ---------------------------------------------------------------------------
# Tests: _MOOD_INTENSITY covers all non-neutral emotions
# ---------------------------------------------------------------------------


def test_mood_intensity_has_all_non_neutral_emotions() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = time.time()

    for emotion in _NON_NEUTRAL_EMOTIONS:
        assert emotion in agent._MOOD_INTENSITY, f"Missing '{emotion}' in _MOOD_INTENSITY"


def test_mood_intensity_all_values_in_range() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = time.time()

    for emotion, intensity in agent._MOOD_INTENSITY.items():
        assert 0.0 < intensity <= 1.0, f"Invalid intensity {intensity} for '{emotion}'"


# ---------------------------------------------------------------------------
# Tests: _interoception agent mood feels covers all non-neutral emotions
# ---------------------------------------------------------------------------


def test_interoception_has_feel_for_all_non_neutral_moods() -> None:
    for emotion in _NON_NEUTRAL_EMOTIONS:
        result = _interoception(
            started_at=time.time() - 60,
            turn_count=1,
            companion_mood="engaged",
            agent_mood=emotion,
            agent_mood_intensity=0.7,
        )
        # Should produce a non-empty result that contains something about mood
        assert result, f"_interoception produced empty result for agent_mood='{emotion}'"


def test_interoception_new_emotions_produce_distinct_feels() -> None:
    """relieved, tender, playful, proud should each produce unique mood feel text."""
    results = {}
    for emotion in ("relieved", "tender", "playful", "proud"):
        results[emotion] = _interoception(
            started_at=time.time() - 60,
            turn_count=1,
            companion_mood="engaged",
            agent_mood=emotion,
            agent_mood_intensity=0.7,
        )
    # All 4 should be different strings
    unique_results = set(results.values())
    assert len(unique_results) == 4, "New emotions should produce distinct interoception text"


# ---------------------------------------------------------------------------
# Tests: _update_mood handles all new emotions
# ---------------------------------------------------------------------------


def test_update_mood_handles_all_new_emotions() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = time.time()

    for emotion in ("relieved", "tender", "playful", "proud"):
        agent._mood = "neutral"
        agent._mood_intensity = 0.0
        agent._update_mood(emotion)
        assert agent._mood == emotion, f"_update_mood failed for '{emotion}'"
        assert agent._mood_intensity > 0.0
