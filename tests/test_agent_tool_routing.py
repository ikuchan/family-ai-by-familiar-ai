"""Tests for EmbodiedAgent._execute_tool() routing logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.exploration import ExplorationTracker


# ---------------------------------------------------------------------------
# Minimal agent factory
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    with_camera: bool = False,
    with_mobility: bool = False,
    with_tts: bool = False,
    with_mcp: bool = False,
):
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.config = MagicMock()
    agent._exploration = ExplorationTracker()
    agent._scene = None

    mem_tool = MagicMock()
    mem_tool.call = AsyncMock(return_value=("mem result", None))
    agent._memory_tool = mem_tool

    coding = MagicMock()
    coding.call = AsyncMock(return_value=("code result", None))
    agent._coding = coding

    agent._camera = None
    agent._mobility = None
    agent._tts = None
    agent._mcp = None

    if with_camera:
        cam = MagicMock()
        cam.call = AsyncMock(return_value=("I see a room", "base64img"))
        agent._camera = cam

    if with_mobility:
        mob = MagicMock()
        mob.call = AsyncMock(return_value=("Moving forward.", None))
        agent._mobility = mob

    if with_tts:
        tts = MagicMock()
        tts.call = AsyncMock(return_value=("Said: hello", None))
        agent._tts = tts

    if with_mcp:
        mcp = MagicMock()
        mcp.call = AsyncMock(return_value=("mcp result", None))
        agent._mcp = mcp

    return agent


# ---------------------------------------------------------------------------
# Tests: memory tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_routes_remember_to_memory_tool():
    agent = _make_agent()
    result, _ = await agent._execute_tool("remember", {"content": "test"})
    assert result == "mem result"
    agent._memory_tool.call.assert_awaited_once_with("remember", {"content": "test"})


@pytest.mark.asyncio
async def test_execute_tool_routes_recall_to_memory_tool():
    agent = _make_agent()
    result, _ = await agent._execute_tool("recall", {"query": "cats"})
    assert result == "mem result"
    agent._memory_tool.call.assert_awaited_once_with("recall", {"query": "cats"})


# ---------------------------------------------------------------------------
# Tests: ToM tool
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: coding tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["read_file", "edit_file", "glob", "grep", "bash"])
async def test_execute_tool_routes_coding_tools(tool_name: str):
    agent = _make_agent()
    result, _ = await agent._execute_tool(tool_name, {})
    assert result == "code result"
    agent._coding.call.assert_awaited_once_with(tool_name, {})


# ---------------------------------------------------------------------------
# Tests: camera tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_routes_see_to_camera():
    agent = _make_agent(with_camera=True)
    result, img = await agent._execute_tool("see", {})
    assert result == "I see a room"
    assert img == "base64img"
    agent._camera.call.assert_awaited_once_with("see", {})


@pytest.mark.asyncio
async def test_execute_tool_camera_missing_returns_mcp_fallback():
    """When camera is absent, 'see' falls through to MCP (if available)."""
    agent = _make_agent(with_mcp=True)
    result, _ = await agent._execute_tool("see", {})
    assert result == "mcp result"
    agent._mcp.call.assert_awaited_once_with("see", {})


@pytest.mark.asyncio
async def test_execute_tool_look_does_not_track_relative_moves():
    """`look` は定点へ絶対移動する。

    `ExplorationTracker` は相対移動の積算（左へ何度、右へ何度）を追う器で、定点とは前提が
    違う。どこを見たかは「見た印」として O に残るので、ここでは繋がない。器そのものの撤去は
    #12（旧系統の撤去）で、旧 `run()` のプロンプトごと落とす。
    """
    agent = _make_agent(with_camera=True)
    agent._camera.call = AsyncMock(return_value=("窓側のほうを向いた。", None))

    before = len(agent._exploration._records)
    await agent._execute_tool("look", {"pose": "窓側"})

    assert len(agent._exploration._records) == before


# ---------------------------------------------------------------------------
# Tests: mobility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_routes_walk_to_mobility():
    agent = _make_agent(with_mobility=True)
    result, _ = await agent._execute_tool("walk", {"direction": "forward"})
    assert result == "Moving forward."
    agent._mobility.call.assert_awaited_once_with("walk", {"direction": "forward"})


# ---------------------------------------------------------------------------
# Tests: TTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_routes_say_to_tts():
    agent = _make_agent(with_tts=True)
    result, _ = await agent._execute_tool("say", {"text": "hello"})
    assert result == "Said: hello"
    agent._tts.call.assert_awaited_once_with("say", {"text": "hello"})


# ---------------------------------------------------------------------------
# Tests: MCP fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_unknown_falls_to_mcp():
    agent = _make_agent(with_mcp=True)
    result, _ = await agent._execute_tool("custom_mcp_tool", {"x": 1})
    assert result == "mcp result"
    agent._mcp.call.assert_awaited_once_with("custom_mcp_tool", {"x": 1})


@pytest.mark.asyncio
async def test_execute_tool_unknown_without_mcp_returns_error():
    """Unknown tool with no MCP configured returns an error message."""
    agent = _make_agent()
    result, img = await agent._execute_tool("nonexistent_tool", {})
    assert "not available" in result.lower() or "nonexistent_tool" in result
    assert img is None
