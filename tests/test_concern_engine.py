"""Tests for lightweight active concerns."""

from __future__ import annotations

from familiar_agent.concern_engine import ConcernEngine
from familiar_agent.prediction import PredictionSignal


def test_prompt_context_respects_turn_cooldown():
    engine = ConcernEngine()
    engine.activate(
        "Something about this still feels unfinished.",
        category="affect",
        intensity=0.75,
        turn_index=5,
    )

    first = engine.context_for_prompt(turn_index=5)
    second = engine.context_for_prompt(turn_index=5)
    third = engine.context_for_prompt(turn_index=7)

    assert first is not None
    assert second is None
    assert third is not None


def test_update_from_turn_keeps_high_salience_threads():
    engine = ConcernEngine()
    signal = PredictionSignal(
        total_error=0.8,
        external_surprise=0.2,
        agency_error=0.7,
        action_name="look",
        observed_entities=("desk",),
        previous_entities=("window",),
        change_ratio=0.1,
    )

    engine.update_from_turn(
        turn_index=3,
        emotion="tender",
        companion_mood="frustrated",
        curiosity="I still want to know what changed near the window.",
        prediction_signal=signal,
    )

    concerns = engine.snapshot()

    assert concerns
    assert any(item["category"] == "agency" for item in concerns)
    assert any(item["category"] == "companion" for item in concerns)


def test_companion_concern_uses_provided_name():
    engine = ConcernEngine()
    engine.update_from_turn(
        turn_index=1,
        emotion="neutral",
        companion_mood="frustrated",
        curiosity=None,
        prediction_signal=None,
        companion_name="Yusuke",
    )
    concerns = engine.snapshot()
    companion = next(c for c in concerns if c["category"] == "companion")
    assert "Yusuke" in companion["topic"]
    assert "Kouta" not in companion["topic"]


def test_companion_concern_falls_back_when_name_empty():
    engine = ConcernEngine()
    engine.update_from_turn(
        turn_index=1,
        emotion="neutral",
        companion_mood="frustrated",
        curiosity=None,
        prediction_signal=None,
        companion_name="",
    )
    concerns = engine.snapshot()
    companion = next(c for c in concerns if c["category"] == "companion")
    assert "companion" in companion["topic"]
    assert "Kouta" not in companion["topic"]


def test_companion_concern_no_name_arg_still_works():
    engine = ConcernEngine()
    engine.update_from_turn(
        turn_index=1,
        emotion="neutral",
        companion_mood="frustrated",
        curiosity=None,
        prediction_signal=None,
    )
    concerns = engine.snapshot()
    assert any(c["category"] == "companion" for c in concerns)
