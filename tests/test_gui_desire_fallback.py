"""Tests for GUI fallback-text suppression on autonomous desire turns.

Step 1 of the silence-control refactor: when a turn is an autonomous desire
turn (desire_name is set) and the model never called say(), the raw fallback
text must NOT be painted into the chat log. Raw text on desire turns leaks
markers like '（内的衝動に従って行動）', URL fragments, and MCP error strings.

On user turns (desire_name == ""), the existing fallback display behaviour is
preserved.
"""

from __future__ import annotations

from familiar_agent.gui import FamiliarWindow


class TestShouldShowAgentFallback:
    def test_user_turn_with_text_shows_fallback(self):
        """User turn (no desire_name) with non-empty text → display it."""
        assert FamiliarWindow._should_show_agent_fallback("こんにちは", "") is True

    def test_desire_turn_with_text_suppresses_fallback(self):
        """Autonomous desire turn → never paint raw fallback text."""
        assert (
            FamiliarWindow._should_show_agent_fallback("（内的衝動に従って行動）", "browse_curiosity")
            is False
        )

    def test_user_turn_with_empty_text_suppresses_fallback(self):
        """Empty text is never displayed, even on a user turn."""
        assert FamiliarWindow._should_show_agent_fallback("", "") is False

    def test_desire_turn_with_empty_text_suppresses_fallback(self):
        assert FamiliarWindow._should_show_agent_fallback("", "share_memory") is False
