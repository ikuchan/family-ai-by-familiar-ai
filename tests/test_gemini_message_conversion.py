"""Tests for GeminiBackend.convert_messages_to_gemini_format().

Verifies that Anthropic-format messages can be explicitly converted to Gemini
format without triggering the "non-Gemini message coerced" warning.
"""

from __future__ import annotations

import logging


from familiar_agent.backends import GeminiBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_user(text: str) -> dict:
    """Anthropic-format user message with string content."""
    return {"role": "user", "content": text}


def _anthropic_user_blocks(blocks: list[dict]) -> dict:
    """Anthropic-format user message with content block list."""
    return {"role": "user", "content": blocks}


def _anthropic_assistant(text: str) -> dict:
    """Anthropic-format assistant message."""
    return {"role": "assistant", "content": text}


def _gemini_user(text: str) -> dict:
    """Gemini-format user message."""
    return {"role": "user", "parts": [{"text": text}]}


def _gemini_model(text: str) -> dict:
    """Gemini-format model (assistant) message."""
    return {"role": "model", "parts": [{"text": text}]}


# ---------------------------------------------------------------------------
# convert_messages_to_gemini_format — unit tests
# ---------------------------------------------------------------------------


def test_to_gemini_message_still_logs_warning_for_accidental_leakage(caplog):
    """_to_gemini_message (the original path) still warns — regression guard."""
    with caplog.at_level(logging.WARNING, logger="familiar_agent.backend"):
        GeminiBackend._to_gemini_message({"role": "user", "content": "oops"})
    assert "coerced" in caplog.text
