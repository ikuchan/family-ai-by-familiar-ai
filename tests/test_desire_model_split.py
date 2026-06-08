"""Tests for is_social_desire() — routing desires to correct LLM backend."""

from __future__ import annotations

from familiar_agent.desires import is_social_desire, _SOCIAL_DESIRE_NAMES


def test_greet_companion_is_social():
    assert is_social_desire("greet_companion") is True


def test_worry_companion_is_social():
    assert is_social_desire("worry_companion") is True


def test_share_memory_is_social():
    assert is_social_desire("share_memory") is True


def test_attachment_is_social():
    assert is_social_desire("attachment") is True


def test_care_is_social():
    assert is_social_desire("care") is True


def test_repair_is_social():
    assert is_social_desire("repair") is True


def test_look_around_is_not_social():
    assert is_social_desire("look_around") is False


def test_explore_is_not_social():
    assert is_social_desire("explore") is False


def test_consolidate_is_not_social():
    assert is_social_desire("consolidate") is False


def test_reflect_is_not_social():
    assert is_social_desire("reflect") is False


def test_browse_curiosity_is_not_social():
    assert is_social_desire("browse_curiosity") is False


def test_rest_is_not_social():
    assert is_social_desire("rest") is False


def test_curiosity_is_not_social():
    assert is_social_desire("curiosity") is False


def test_unknown_desire_is_not_social():
    assert is_social_desire("unknown_drive_xyz") is False


def test_social_desire_names_is_frozenset():
    assert isinstance(_SOCIAL_DESIRE_NAMES, frozenset)


def test_social_names_subset_consistency():
    # Every name in _SOCIAL_DESIRE_NAMES must be recognized as social
    for name in _SOCIAL_DESIRE_NAMES:
        assert is_social_desire(name) is True, f"{name} should be social"
