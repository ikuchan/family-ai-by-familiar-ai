"""Tests for DeferredSearchTool background search."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from familiar_agent.tools.deferred_search import DeferredSearchTool


def _make_tool(result: str = "search result") -> tuple[DeferredSearchTool, AsyncMock]:
    fn = AsyncMock(return_value=(result, None))
    return DeferredSearchTool(fn), fn


@pytest.mark.asyncio
async def test_call_returns_immediately_without_waiting():
    tool, fn = _make_tool()
    result, _ = await tool.call("search_deferred", {"query": "test query"})
    assert "test query" in result
    assert "バックグラウンド" in result


@pytest.mark.asyncio
async def test_pending_context_empty_before_completion():
    tool, _ = _make_tool()
    await tool.call("search_deferred", {"query": "test"})
    # Before background task runs
    assert tool.pending_context() == ""


@pytest.mark.asyncio
async def test_pending_context_contains_result_after_completion():
    tool, _ = _make_tool("大阪城は17世紀に建てられた。")
    await tool.call("search_deferred", {"query": "大阪城"})
    await asyncio.sleep(0)  # yield to let background task run
    ctx = tool.pending_context()
    assert "大阪城" in ctx
    assert "大阪城は17世紀に建てられた。" in ctx


@pytest.mark.asyncio
async def test_pending_context_cleared_after_read():
    tool, _ = _make_tool("result")
    await tool.call("search_deferred", {"query": "query"})
    await asyncio.sleep(0)
    tool.pending_context()  # consume
    assert tool.pending_context() == ""


@pytest.mark.asyncio
async def test_uses_brave_by_default():
    tool, fn = _make_tool()
    await tool.call("search_deferred", {"query": "hello"})
    await asyncio.sleep(0)
    fn.assert_called_once()
    assert fn.call_args[0][0] == "brave_web_search"


@pytest.mark.asyncio
async def test_uses_tavily_when_specified():
    tool, fn = _make_tool()
    await tool.call("search_deferred", {"query": "hello", "source": "tavily"})
    await asyncio.sleep(0)
    fn.assert_called_once()
    assert fn.call_args[0][0] == "tavily_search"


@pytest.mark.asyncio
async def test_empty_query_returns_error():
    tool, fn = _make_tool()
    result, _ = await tool.call("search_deferred", {"query": ""})
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
    assert defs[0]["name"] == "search_deferred"
    assert "input_schema" in defs[0]
    assert "query" in defs[0]["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_concurrent_limit_blocks_excess_calls():
    tool, _ = _make_tool()
    tool._running = 3  # simulate max concurrent
    result, _ = await tool.call("search_deferred", {"query": "overflow"})
    assert "混んでいます" in result


@pytest.mark.asyncio
async def test_search_error_stored_as_pending_result():
    fn = AsyncMock(side_effect=RuntimeError("network error"))
    tool = DeferredSearchTool(fn)
    await tool.call("search_deferred", {"query": "fail query"})
    await asyncio.sleep(0)
    ctx = tool.pending_context()
    assert "エラー" in ctx
