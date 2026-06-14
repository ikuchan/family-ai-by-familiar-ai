"""Tests for internal-desire-turn say() suppression (Issue D emergency fix).

Internal desire turns (look_around etc.) must NOT call TTS — they bypass the
presence/quiet gate.  Social desires and user turns continue to speak normally.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture()
def agent(monkeypatch):
    """Minimal EmbodiedAgent with no external hardware dependencies."""
    monkeypatch.setenv("API_KEY", "sk-test-dummy")
    monkeypatch.setenv("PLATFORM", "anthropic")
    monkeypatch.setenv("CAMERA_HOST", "")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.setenv("TUYA_API_KEY", "")
    monkeypatch.setenv("MCP_CONFIG", "")

    from familiar_agent.config import AgentConfig
    from familiar_agent.agent import EmbodiedAgent

    config = AgentConfig()
    return EmbodiedAgent(config)


@pytest.mark.asyncio
async def test_say_suppressed_on_internal_desire_turn(agent):
    """Internal desire turns: say() is blocked and TTS is never called."""
    agent._current_is_desire_turn = True
    agent._current_desire_name = "look_around"   # internal (not social)
    agent._tts = MagicMock()
    agent._tts.call = AsyncMock(return_value=("(silent) x", None))

    text, img = await agent._execute_tool("say", {"text": "見えるよ"})

    assert "suppress" in text.lower()
    assert img is None
    agent._tts.call.assert_not_called()


@pytest.mark.asyncio
async def test_say_allowed_on_social_desire_turn(agent):
    """Social desire turns (share_memory etc.): say() executes normally."""
    agent._current_is_desire_turn = True
    agent._current_desire_name = "share_memory"  # social
    agent._tts = MagicMock()
    agent._tts.call = AsyncMock(return_value=("ok", None))

    await agent._execute_tool("say", {"text": "思い出したよ"})

    agent._tts.call.assert_called_once()


@pytest.mark.asyncio
async def test_say_allowed_on_user_turn(agent):
    """User turns: say() executes normally regardless of desire_name."""
    agent._current_is_desire_turn = False
    agent._current_desire_name = ""
    agent._tts = MagicMock()
    agent._tts.call = AsyncMock(return_value=("ok", None))

    await agent._execute_tool("say", {"text": "こんにちは"})

    agent._tts.call.assert_called_once()
