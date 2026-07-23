"""Tests for GeminiBackend.convert_messages_to_gemini_format().

Verifies that Anthropic-format messages can be explicitly converted to Gemini
format without triggering the "non-Gemini message coerced" warning.
"""

from __future__ import annotations

import logging


from familiar_agent.backend import GeminiBackend


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


def test_converts_string_content_user_message():
    msgs = [_anthropic_user("hello")]
    result = GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert result == [_gemini_user("hello")]


def test_converts_assistant_role_to_model():
    msgs = [_anthropic_assistant("hi there")]
    result = GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert result == [_gemini_model("hi there")]


def test_converts_list_content_with_text_blocks():
    msgs = [_anthropic_user_blocks([{"type": "text", "text": "block content"}])]
    result = GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert result == [_gemini_user("block content")]


def test_converts_list_content_skips_non_text_blocks():
    msgs = [_anthropic_user_blocks([
        {"type": "tool_result", "content": "ignored"},
        {"type": "text", "text": "kept"},
    ])]
    result = GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert result[0]["parts"] == [{"text": "kept"}]


def test_already_gemini_format_passes_through_unchanged():
    already_gemini = {"role": "user", "parts": [{"text": "native"}]}
    result = GeminiBackend.convert_messages_to_gemini_format([already_gemini])
    assert result == [already_gemini]


def test_converts_mixed_list_of_messages():
    msgs = [
        _anthropic_user("first"),
        _anthropic_assistant("second"),
        _anthropic_user("third"),
    ]
    result = GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert result == [
        _gemini_user("first"),
        _gemini_model("second"),
        _gemini_user("third"),
    ]


def test_does_not_log_warning(caplog):
    """convert_messages_to_gemini_format must NOT emit the coercion warning."""
    msgs = [_anthropic_user("test"), _anthropic_assistant("reply")]
    with caplog.at_level(logging.WARNING, logger="familiar_agent.backend"):
        GeminiBackend.convert_messages_to_gemini_format(msgs)
    assert "coerced" not in caplog.text


def test_to_gemini_message_still_logs_warning_for_accidental_leakage(caplog):
    """_to_gemini_message (the original path) still warns — regression guard."""
    with caplog.at_level(logging.WARNING, logger="familiar_agent.backend"):
        GeminiBackend._to_gemini_message({"role": "user", "content": "oops"})
    assert "coerced" in caplog.text
