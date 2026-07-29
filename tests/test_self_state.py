"""Tests for persistent latent self state."""

from __future__ import annotations


from familiar_agent.self_state import SelfState
from familiar_agent.workspace import Coalition


def _coalition(source: str, *, activation: float = 0.8, urgency: float = 0.5, novelty: float = 0.4):
    return Coalition(
        source=source,
        summary=f"{source} event",
        activation=activation,
        urgency=urgency,
        novelty=novelty,
        context_block=f"[{source}]",
    )


def test_prediction_broadcast_raises_arousal_and_tension():
    state = SelfState()
    before = state.snapshot()

    state.apply_broadcast(_coalition("prediction", novelty=0.9, urgency=0.8))
    after = state.snapshot()

    assert after["arousal"] > before["arousal"]
    assert after["unresolved_tension"] > before["unresolved_tension"]
    assert after["sensor_confidence"] < before["sensor_confidence"]


def test_memory_broadcast_strengthens_social_pull_and_focus():
    state = SelfState()
    before = state.snapshot()

    state.apply_broadcast(_coalition("memory", activation=0.9))
    after = state.snapshot()

    assert after["social_pull"] > before["social_pull"]
    assert after["focus_stability"] > before["focus_stability"]
    assert after["unresolved_tension"] < before["unresolved_tension"]


def test_default_mode_broadcast_settles_arousal():
    state = SelfState()
    state.apply_broadcast(_coalition("prediction", novelty=1.0, urgency=1.0))
    activated = state.snapshot()["arousal"]

    state.apply_broadcast(_coalition("default_mode", activation=0.5, urgency=0.1, novelty=0.0))
    settled = state.snapshot()["arousal"]

    assert settled < activated


def test_state_persists_across_instances():
    state = SelfState()
    state.apply_broadcast(_coalition("memory", activation=0.95))

    reloaded = SelfState()
    assert reloaded.snapshot() == state.snapshot()


def test_successful_action_prediction_increases_sensor_confidence():
    state = SelfState()
    before = state.snapshot()

    state.apply_prediction_feedback(
        external_surprise=0.1,
        agency_error=0.0,
        action_name="look",
    )
    after = state.snapshot()

    assert after["sensor_confidence"] > before["sensor_confidence"]


def test_agency_error_reduces_sensor_confidence():
    state = SelfState()
    before = state.snapshot()

    state.apply_prediction_feedback(
        external_surprise=0.1,
        agency_error=0.8,
        action_name="walk",
    )
    after = state.snapshot()

    assert after["sensor_confidence"] < before["sensor_confidence"]
    assert after["unresolved_tension"] > before["unresolved_tension"]


def test_turn_context_drifts_state_toward_baseline():
    state = SelfState()
    state.apply_broadcast(_coalition("prediction", novelty=1.0, urgency=1.0))
    activated = state.snapshot()

    state.apply_turn_context(emotion="neutral", companion_mood="absent")
    settled = state.snapshot()

    assert settled["arousal"] < activated["arousal"]


def test_turn_context_carries_social_concern_forward():
    state = SelfState()
    before = state.snapshot()

    state.apply_turn_context(
        emotion="tender",
        companion_mood="frustrated",
        curiosity="I still want to understand what Kouta noticed.",
    )
    after = state.snapshot()

    assert after["social_pull"] > before["social_pull"]
    assert after["unresolved_tension"] > before["unresolved_tension"]
    assert after["focus_stability"] > before["focus_stability"]


