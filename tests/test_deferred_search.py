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
async def test_blocks_when_at_max_concurrent():
    from familiar_agent.tools.deferred_search import _MAX_CONCURRENT

    tool, _ = _make_tool()
    tool._running = _MAX_CONCURRENT  # simulate max in-flight searches
    result, _ = await tool.call("search_deferred", {"query": "overflow"})
    assert "同時に" in result


@pytest.mark.asyncio
async def test_does_not_block_when_below_max_concurrent():
    from familiar_agent.tools.deferred_search import _MAX_CONCURRENT

    tool, _ = _make_tool()
    tool._running = _MAX_CONCURRENT - 1
    result, _ = await tool.call("search_deferred", {"query": "under limit"})
    assert "バックグラウンド" in result


@pytest.mark.asyncio
async def test_does_not_block_when_pending_result_exists():
    tool, fn = _make_tool()
    tool._pending = [{"query": "prev", "result": "r", "source": "brave"}]
    result, _ = await tool.call("search_deferred", {"query": "new query"})
    assert "バックグラウンド" in result
    await asyncio.sleep(0)
    fn.assert_called_once()


@pytest.mark.asyncio
async def test_search_error_stored_as_pending_result():
    fn = AsyncMock(side_effect=RuntimeError("network error"))
    tool = DeferredSearchTool(fn)
    await tool.call("search_deferred", {"query": "fail query"})
    await asyncio.sleep(0)
    ctx = tool.pending_context()
    assert "エラー" in ctx


# ── Deduplication tests ───────────────────────────────────────────────────────


class _MockUtilityBackend:
    """Minimal utility backend stub that always returns a fixed yes/no answer."""

    def __init__(self, response: str = "no") -> None:
        self.response = response
        self.calls: list[str] = []

    async def complete(self, prompt: str, max_tokens: int, *,
                       system: str | None = None) -> str:
        self.calls.append(prompt)
        return self.response


@pytest.mark.asyncio
async def test_first_search_skips_utility_llm():
    """When no existing searches exist, utility LLM is never called."""
    backend = _MockUtilityBackend("yes")
    tool = DeferredSearchTool(AsyncMock(return_value=("r", None)), utility_backend=backend)
    result, _ = await tool.call("search_deferred", {"query": "first query"})
    assert "バックグラウンド" in result
    assert len(backend.calls) == 0


@pytest.mark.asyncio
async def test_same_intent_blocked_by_utility_llm():
    """Utility LLM returning yes blocks the duplicate search."""
    backend = _MockUtilityBackend("yes")
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn, utility_backend=backend)

    await tool.call("search_deferred", {"query": "日本 最新ニュース"})
    result, _ = await tool.call("search_deferred", {"query": "今日のニュース教えて"})

    assert "進行中" in result
    await asyncio.sleep(0)
    assert fn.call_count == 1  # only the first search executed


@pytest.mark.asyncio
async def test_different_intent_allowed_by_utility_llm():
    """Utility LLM returning no allows the second search to proceed."""
    backend = _MockUtilityBackend("no")
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn, utility_backend=backend)

    await tool.call("search_deferred", {"query": "日本 最新ニュース"})
    result, _ = await tool.call("search_deferred", {"query": "東京 天気予報"})

    assert "バックグラウンド" in result
    await asyncio.sleep(0)
    assert fn.call_count == 2


@pytest.mark.asyncio
async def test_no_utility_backend_exact_match_blocks_duplicate():
    """Without utility backend, exact string match blocks duplicate."""
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn)

    await tool.call("search_deferred", {"query": "same query"})
    result, _ = await tool.call("search_deferred", {"query": "same query"})
    assert "進行中" in result


@pytest.mark.asyncio
async def test_no_utility_backend_different_query_allowed():
    """Without utility backend, different query string proceeds."""
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn)

    await tool.call("search_deferred", {"query": "query A"})
    result, _ = await tool.call("search_deferred", {"query": "query B"})
    assert "バックグラウンド" in result


@pytest.mark.asyncio
async def test_running_query_tracked_and_removed():
    """Query is in _running_queries while executing and removed on completion."""
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn)

    await tool.call("search_deferred", {"query": "track me"})
    assert "track me" in tool._running_queries
    await asyncio.sleep(0)
    assert "track me" not in tool._running_queries


@pytest.mark.asyncio
async def test_pending_query_also_blocks_duplicate():
    """A query already delivered to _pending blocks the same query again."""
    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn)
    tool._pending = [{"query": "cached query", "result": "...", "source": "brave"}]

    result, _ = await tool.call("search_deferred", {"query": "cached query"})
    assert "進行中" in result
    fn.assert_not_called()


@pytest.mark.asyncio
async def test_utility_llm_error_falls_back_to_exact_match():
    """If utility LLM raises, exact match is used as fallback."""

    class _BrokenBackend:
        async def complete(self, prompt: str, max_tokens: int, *,
                       system: str | None = None) -> str:
            raise RuntimeError("LLM unavailable")

    fn = AsyncMock(return_value=("r", None))
    tool = DeferredSearchTool(fn, utility_backend=_BrokenBackend())

    await tool.call("search_deferred", {"query": "identical"})
    result, _ = await tool.call("search_deferred", {"query": "identical"})
    assert "進行中" in result


# ---------------------------------------------------------------------------
# user_initiated flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_has_user_initiated_pending_false_by_default():
    tool, _ = _make_tool()
    await tool.call("search_deferred", {"query": "q"})
    await asyncio.sleep(0)
    assert tool.has_user_initiated_pending is False


@pytest.mark.asyncio
async def test_has_user_initiated_pending_true_when_user_turn():
    tool, _ = _make_tool()
    tool.set_user_turn(True)
    await tool.call("search_deferred", {"query": "q"})
    await asyncio.sleep(0)
    assert tool.has_user_initiated_pending is True


@pytest.mark.asyncio
async def test_has_user_initiated_pending_false_when_desire_turn():
    tool, _ = _make_tool()
    tool.set_user_turn(False)
    await tool.call("search_deferred", {"query": "q"})
    await asyncio.sleep(0)
    assert tool.has_user_initiated_pending is False


@pytest.mark.asyncio
async def test_has_user_initiated_pending_cleared_after_read():
    tool, _ = _make_tool()
    tool.set_user_turn(True)
    await tool.call("search_deferred", {"query": "q"})
    await asyncio.sleep(0)
    tool.pending_context()  # consume
    assert tool.has_user_initiated_pending is False


# ── Tool description contract ──────────────────────────────────────────────────


def test_deferred_search_tooldef_requires_say_in_own_words() -> None:
    """Tool description must instruct the model to report via say() in own words."""
    tool, _ = _make_tool()
    defs = tool.get_tool_definitions()
    desc = defs[0]["description"]
    # Must mention say() so the model knows to speak the result aloud
    assert "say()" in desc
    # Must mention reporting in own words (not raw source text)
    assert any(kw in desc for kw in ("自分の言葉", "口語", "だよ", "みたい"))
