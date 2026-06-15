"""Tests for is_internal_desire_turn() — shared UI display routing helper."""
from __future__ import annotations

from familiar_agent.desires import is_internal_desire_turn


class TestIsInternalDesireTurn:
    def test_user_turn_is_not_internal(self):
        assert is_internal_desire_turn("") is False

    def test_internal_desires_are_internal(self):
        assert is_internal_desire_turn("explore") is True
        assert is_internal_desire_turn("look_around") is True

    def test_social_desires_are_not_internal(self):
        assert is_internal_desire_turn("greet_companion") is False
        assert is_internal_desire_turn("share_memory") is False
        assert is_internal_desire_turn("share_search_result") is False
