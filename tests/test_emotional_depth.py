"""Tests for expanded emotion vocabulary (Phase 5 — emotional depth).

文字列 mood 側（_MOOD_INTENSITY・_update_mood）が
12ラベルを網羅することを確認する。感情ラベルは W2b-2 で PAD から派生する形へ
変わったが、この文字列 mood 機構は移行期も生きている。
"""

from __future__ import annotations

import time

from familiar_agent.agent import EmbodiedAgent


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
# _MOOD_INTENSITY・_update_mood（現存機構）の網羅を引き続き確認する。


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
