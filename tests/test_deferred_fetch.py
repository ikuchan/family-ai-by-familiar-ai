"""Tests for DeferredFetchTool background URL fetch."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from familiar_agent.tools.deferred_fetch import DeferredFetchTool


def _make_tool(result: str = "page content") -> tuple[DeferredFetchTool, AsyncMock]:
    fn = AsyncMock(return_value=(result, None))
    return DeferredFetchTool(fn), fn


@pytest.mark.asyncio
async def test_call_returns_immediately_without_waiting():
    tool, fn = _make_tool()
    result, _ = await tool.call("fetch_deferred", {"url": "https://example.com"})
    assert "https://example.com" in result
    assert "バックグラウンド" in result


@pytest.mark.asyncio
async def test_pending_context_empty_before_completion():
    tool, _ = _make_tool()
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    assert tool.pending_context() == ""


@pytest.mark.asyncio
async def test_pending_context_contains_result_after_completion():
    tool, _ = _make_tool("東京は日本の首都です。")
    await tool.call("fetch_deferred", {"url": "https://example.com/tokyo"})
    await asyncio.sleep(0)
    ctx = tool.pending_context()
    # URL label is intentionally excluded from context (injected via inner_voice instead).
    assert "東京は日本の首都です。" in ctx


@pytest.mark.asyncio
async def test_pending_summary_returns_urls_without_clearing():
    tool, _ = _make_tool("result")
    await tool.call("fetch_deferred", {"url": "https://example.com/a"})
    await asyncio.sleep(0)
    summary = tool.pending_summary()
    assert "https://example.com/a" in summary
    assert tool.has_pending  # pending list was not cleared


@pytest.mark.asyncio
async def test_pending_context_cleared_after_read():
    tool, _ = _make_tool("result")
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await asyncio.sleep(0)
    tool.pending_context()  # consume
    assert tool.pending_context() == ""


@pytest.mark.asyncio
async def test_calls_fetch_mcp_tool():
    tool, fn = _make_tool()
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await asyncio.sleep(0)
    fn.assert_called_once()
    assert fn.call_args[0][0] == "fetch"
    assert fn.call_args[0][1] == {"url": "https://example.com"}


@pytest.mark.asyncio
async def test_empty_url_returns_error():
    tool, fn = _make_tool()
    result, _ = await tool.call("fetch_deferred", {"url": ""})
    assert "空" in result
    fn.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_tool_name_returns_error():
    tool, _ = _make_tool()
    result, _ = await tool.call("unknown", {})
    assert "Unknown" in result


@pytest.mark.asyncio
async def test_get_tool_definitions_valid():
    tool, _ = _make_tool()
    defs = tool.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "fetch_deferred"
    assert "input_schema" in defs[0]
    assert "url" in defs[0]["input_schema"]["properties"]
    assert defs[0]["input_schema"]["required"] == ["url"]


@pytest.mark.asyncio
async def test_blocks_when_at_max_concurrent():
    from familiar_agent.tools.deferred_fetch import _MAX_CONCURRENT

    tool, _ = _make_tool()
    tool._running = _MAX_CONCURRENT  # simulate max in-flight fetches
    result, _ = await tool.call("fetch_deferred", {"url": "https://example.com"})
    assert "同時に" in result


@pytest.mark.asyncio
async def test_does_not_block_when_below_max_concurrent():
    from familiar_agent.tools.deferred_fetch import _MAX_CONCURRENT

    tool, _ = _make_tool()
    tool._running = _MAX_CONCURRENT - 1
    result, _ = await tool.call("fetch_deferred", {"url": "https://example.com"})
    assert "バックグラウンド" in result


@pytest.mark.asyncio
async def test_does_not_block_when_pending_result_exists():
    tool, fn = _make_tool()
    tool._pending = [{"url": "https://prev.com", "result": "r"}]
    result, _ = await tool.call("fetch_deferred", {"url": "https://example.com"})
    assert "バックグラウンド" in result
    await asyncio.sleep(0)
    fn.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_error_stored_as_pending_result():
    fn = AsyncMock(side_effect=RuntimeError("connection error"))
    tool = DeferredFetchTool(fn)
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await asyncio.sleep(0)
    ctx = tool.pending_context()
    assert "エラー" in ctx


@pytest.mark.asyncio
async def test_is_running_true_while_task_in_flight():
    import asyncio as _asyncio

    barrier = _asyncio.Event()

    async def slow_fetch(tool_name: str, tool_input: dict):
        await barrier.wait()
        return "done", None

    tool = DeferredFetchTool(slow_fetch)
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await _asyncio.sleep(0)
    assert tool.is_running is True
    barrier.set()
    await _asyncio.sleep(0)
    assert tool.is_running is False


@pytest.mark.asyncio
async def test_has_pending_false_while_running():
    import asyncio as _asyncio

    barrier = _asyncio.Event()

    async def slow_fetch(tool_name: str, tool_input: dict):
        await barrier.wait()
        return "done", None

    tool = DeferredFetchTool(slow_fetch)
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await _asyncio.sleep(0)
    assert tool.has_pending is False
    barrier.set()
    await _asyncio.sleep(0)
    assert tool.has_pending is True


@pytest.mark.asyncio
async def test_result_truncated_to_3000_chars():
    long_content = "x" * 5000
    tool, _ = _make_tool(long_content)
    await tool.call("fetch_deferred", {"url": "https://example.com"})
    await asyncio.sleep(0)
    ctx = tool.pending_context()
    assert len(ctx) < 4000  # header + truncated content
