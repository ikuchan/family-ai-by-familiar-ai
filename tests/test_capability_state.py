"""Tests for capability_state — manifest loader and AI self-understanding storage."""

from __future__ import annotations

from familiar_agent.capability_state import (
    load_manifest,
    load_summary,
    save_summary,
    should_refresh,
)


def test_load_manifest_returns_yaml_string():
    text = load_manifest()
    assert isinstance(text, str)
    assert len(text) > 0
    assert "capabilities" in text


def test_load_manifest_contains_known_capability():
    text = load_manifest()
    assert "autonomous_initiation" in text
    assert "memory" in text


def test_save_and_load_summary_roundtrip():
    save_summary("- I can recall memories.\n- I can speak autonomously.")
    result = load_summary()
    assert "recall memories" in result
    assert "autonomously" in result


def test_save_summary_overwrites_previous():
    save_summary("first summary")
    save_summary("second summary")
    result = load_summary()
    assert "second summary" in result
    assert "first summary" not in result


def test_load_summary_returns_empty_when_missing():
    # After truncation by conftest, agent_state is empty
    result = load_summary()
    assert result == ""


def test_should_refresh_on_turn_zero_when_no_summary():
    assert should_refresh(0) is True


def test_should_refresh_false_on_turn_zero_when_summary_exists():
    save_summary("existing summary")
    assert should_refresh(0) is False


def test_should_refresh_on_multiples_of_50():
    save_summary("existing summary")
    assert should_refresh(50) is True
    assert should_refresh(100) is True
    assert should_refresh(150) is True


def test_should_not_refresh_on_other_turns():
    save_summary("existing summary")
    for turn in [1, 10, 25, 49, 51, 99]:
        assert should_refresh(turn) is False
