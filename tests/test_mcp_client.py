"""Tests for MCPClientManager — stdio + SSE transport, tool routing, error handling.

All tests mock the mcp package so no real MCP server is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AsyncCM:
    """Minimal async context manager that yields a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass


def _tool(name: str, description: str = "", schema: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


def _session(*tools: MagicMock) -> AsyncMock:
    """Return a session mock that lists the given tools."""
    tools_result = MagicMock()
    tools_result.tools = list(tools)
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=tools_result)
    return session


def _patch_mcp(sessions_in_order: list[AsyncMock]) -> dict:
    """
    Build a sys.modules patch dict.  ClientSession returns sessions sequentially.
    """
    call_idx = [0]

    class _ClientSessionCM:
        def __init__(self, *args, **kwargs):
            self._idx = call_idx[0]
            call_idx[0] += 1

        async def __aenter__(self):
            return sessions_in_order[self._idx]

        async def __aexit__(self, *args):
            pass

    rw = (MagicMock(), MagicMock())
    transport_cm = _AsyncCM(rw)

    mcp_mod = MagicMock()
    mcp_mod.ClientSession = _ClientSessionCM
    mcp_mod.StdioServerParameters = MagicMock()

    mcp_stdio = MagicMock()
    mcp_stdio.stdio_client = MagicMock(return_value=transport_cm)

    mcp_sse = MagicMock()
    mcp_sse.sse_client = MagicMock(return_value=transport_cm)

    return {
        "mcp": mcp_mod,
        "mcp.client": MagicMock(),
        "mcp.client.stdio": mcp_stdio,
        "mcp.client.sse": mcp_sse,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_config_file_returns_empty_tools(tmp_path: Path) -> None:
    """MCPClientManager with a non-existent config file exposes no tools."""
    from familiar_agent.mcp_client import MCPClientManager

    mgr = MCPClientManager(config_path=tmp_path / "missing.json")
    await mgr.start()
    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_empty_mcp_servers_returns_empty_tools(tmp_path: Path) -> None:
    """Config with empty mcpServers dict connects to nothing."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))

    from familiar_agent.mcp_client import MCPClientManager

    mgr = MCPClientManager(config_path=cfg)
    await mgr.start()
    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_stdio_server_registers_tools(tmp_path: Path) -> None:
    """stdio server: tools are registered and accessible via get_tool_definitions."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@mcp/server-fs"],
                    }
                }
            }
        )
    )
    sess = _session(_tool("read_file", "Read a file"), _tool("write_file", "Write a file"))

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    defs = mgr.get_tool_definitions()
    names = {d["name"] for d in defs}
    assert names == {"read_file", "write_file"}


@pytest.mark.asyncio
async def test_sse_server_registers_tools(tmp_path: Path) -> None:
    """SSE server: tools are registered via sse_client transport."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "memory": {
                        "type": "sse",
                        "url": "http://localhost:3000/sse",
                    }
                }
            }
        )
    )
    sess = _session(_tool("remember", "Store a memory"), _tool("recall", "Retrieve memories"))

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    defs = mgr.get_tool_definitions()
    names = {d["name"] for d in defs}
    assert names == {"remember", "recall"}


@pytest.mark.asyncio
async def test_unknown_type_is_skipped(tmp_path: Path) -> None:
    """Server with unsupported type is skipped; no crash."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ws": {
                        "type": "websocket",
                        "url": "ws://localhost:9000",
                    }
                }
            }
        )
    )

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, _patch_mcp([])):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_stdio_missing_command_is_skipped(tmp_path: Path) -> None:
    """stdio server without 'command' is skipped gracefully."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"broken": {"type": "stdio", "args": []}}}))

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, _patch_mcp([])):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_sse_missing_url_is_skipped(tmp_path: Path) -> None:
    """SSE server without 'url' is skipped gracefully."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"broken": {"type": "sse"}}}))

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, _patch_mcp([])):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_connection_failure_is_skipped(tmp_path: Path) -> None:
    """If a server raises on connect, it is skipped; no crash."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"bad": {"type": "stdio", "command": "nonexistent"}}}))

    from unittest.mock import patch

    mcp_mod = MagicMock()
    mcp_mod.StdioServerParameters = MagicMock()

    class _BoomCM:
        async def __aenter__(self):
            raise ConnectionRefusedError("Server not found")

        async def __aexit__(self, *args):
            pass

    mcp_stdio = MagicMock()
    mcp_stdio.stdio_client = MagicMock(return_value=_BoomCM())
    # ClientSession should never be reached
    mcp_mod.ClientSession = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mcp": mcp_mod,
            "mcp.client": MagicMock(),
            "mcp.client.stdio": mcp_stdio,
            "mcp.client.sse": MagicMock(),
        },
    ):
        from familiar_agent.mcp_client import MCPClientManager

        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_tool_name_collision_first_wins(tmp_path: Path) -> None:
    """When two servers expose the same tool name, first-registered server wins."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "server_a": {"type": "stdio", "command": "cmd_a"},
                    "server_b": {"type": "stdio", "command": "cmd_b"},
                }
            }
        )
    )
    sess_a = _session(_tool("shared_tool"))
    sess_b = _session(_tool("shared_tool"))

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess_a, sess_b])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    # Only one definition for the shared tool
    defs = mgr.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "shared_tool"


@pytest.mark.asyncio
async def test_mcp_not_installed_returns_empty(tmp_path: Path) -> None:
    """If mcp package is not installed, get_tool_definitions() returns []."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"srv": {"type": "stdio", "command": "cmd"}}}))

    from unittest.mock import patch

    with patch.dict(
        sys.modules,
        {"mcp": None, "mcp.client": None, "mcp.client.stdio": None, "mcp.client.sse": None},
    ):
        from familiar_agent.mcp_client import MCPClientManager

        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()

    assert mgr.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_call_unknown_tool_returns_error(tmp_path: Path) -> None:
    """call() for a tool that was never registered returns an error string, no exception."""
    from familiar_agent.mcp_client import MCPClientManager

    mgr = MCPClientManager(config_path=tmp_path / "missing.json")
    text, image = await mgr.call("nonexistent_tool", {})
    assert "not found" in text.lower() or "nonexistent_tool" in text
    assert image is None


@pytest.mark.asyncio
async def test_call_routes_to_correct_server_and_returns_text(tmp_path: Path) -> None:
    """call() routes to the right session and extracts text from content blocks."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"mem": {"type": "sse", "url": "http://localhost:9000/sse"}}})
    )
    sess = _session(_tool("remember"))

    # Fake call_tool result
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Stored successfully."
    call_result = MagicMock()
    call_result.content = [text_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        text, image = await mgr.call("remember", {"text": "hello"})

    assert text == "Stored successfully."
    assert image is None
    sess.call_tool.assert_awaited_once_with("remember", arguments={"text": "hello"})


@pytest.mark.asyncio
async def test_call_extracts_image_from_content(tmp_path: Path) -> None:
    """call() extracts base64 image data when content includes an image block."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"cam": {"type": "stdio", "command": "cam_cmd"}}}))
    sess = _session(_tool("capture"))

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Image captured."
    img_block = MagicMock()
    img_block.type = "image"
    img_block.data = "base64encodedimagedata=="
    call_result = MagicMock()
    call_result.content = [text_block, img_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        text, image = await mgr.call("capture", {})

    assert text == "Image captured."
    assert image == "base64encodedimagedata=="


@pytest.mark.asyncio
async def test_tavily_search_strips_country_param(tmp_path: Path) -> None:
    """tavily_search calls must have the 'country' key removed before dispatch."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"tavily": {"type": "stdio", "command": "tavily-cmd"}}})
    )
    sess = _session(_tool("tavily_search"))

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Results."
    call_result = MagicMock()
    call_result.content = [text_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        await mgr.call("tavily_search", {"query": "AI news", "country": "Japan", "max_results": 5})

    _, kwargs = sess.call_tool.call_args
    sent = kwargs["arguments"]
    assert "country" not in sent
    assert sent["query"] == "AI news"
    assert sent["max_results"] == 5


@pytest.mark.asyncio
async def test_tavily_search_without_country_is_unchanged(tmp_path: Path) -> None:
    """tavily_search calls without 'country' are passed through unchanged."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"tavily": {"type": "stdio", "command": "tavily-cmd"}}})
    )
    sess = _session(_tool("tavily_search"))

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Results."
    call_result = MagicMock()
    call_result.content = [text_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        await mgr.call("tavily_search", {"query": "AI news", "search_depth": "basic"})

    _, kwargs = sess.call_tool.call_args
    sent = kwargs["arguments"]
    assert sent == {"query": "AI news", "search_depth": "basic"}


@pytest.mark.asyncio
async def test_other_tools_country_param_not_stripped(tmp_path: Path) -> None:
    """'country' is only stripped from tavily_search; other tools are not affected."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"other": {"type": "stdio", "command": "other-cmd"}}})
    )
    sess = _session(_tool("some_tool"))

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "OK."
    call_result = MagicMock()
    call_result.content = [text_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        await mgr.call("some_tool", {"query": "test", "country": "Japan"})

    _, kwargs = sess.call_tool.call_args
    sent = kwargs["arguments"]
    assert "country" in sent


def test_compress_tavily_result_truncates_content() -> None:
    """Content snippets longer than 120 chars are truncated with ellipsis."""
    from familiar_agent.mcp_client import _compress_tavily_result

    long_content = "x" * 200
    raw = f"Detailed Results:\n\nTitle: Test\nURL: https://example.com\nContent: {long_content}\nScore: 0.9\n"
    result = _compress_tavily_result(raw)

    assert "Title: Test" in result
    assert "URL: https://example.com" in result
    assert "Score:" not in result
    content_line = next(ln for ln in result.splitlines() if ln.startswith("Content: "))
    assert content_line.endswith("…")
    assert len(content_line) <= len("Content: ") + 120 + 1  # +1 for ellipsis char


def test_compress_tavily_result_keeps_answer() -> None:
    """Answer: line is preserved as-is (it is already short)."""
    from familiar_agent.mcp_client import _compress_tavily_result

    raw = "Answer: AIによる短い要約文。\n\nDetailed Results:\n\nTitle: T\nURL: U\nContent: short\n"
    result = _compress_tavily_result(raw)
    assert result.startswith("Answer: AIによる短い要約文。")


def test_compress_tavily_result_drops_score_and_favicon() -> None:
    """Score and Favicon lines are removed from output."""
    from familiar_agent.mcp_client import _compress_tavily_result

    raw = "Detailed Results:\n\nTitle: T\nURL: U\nContent: c\nScore: 0.95\nFavicon: https://f.ico\n"
    result = _compress_tavily_result(raw)
    assert "Score:" not in result
    assert "Favicon:" not in result


def test_compress_tavily_result_short_content_unchanged() -> None:
    """Content shorter than 120 chars is passed through without truncation."""
    from familiar_agent.mcp_client import _compress_tavily_result

    raw = "Detailed Results:\n\nTitle: T\nURL: U\nContent: short text\n"
    result = _compress_tavily_result(raw)
    assert "Content: short text" in result
    assert "…" not in result


@pytest.mark.asyncio
async def test_tavily_search_result_is_compressed(tmp_path: Path) -> None:
    """call() applies _compress_tavily_result to tavily_search results."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"tavily": {"type": "stdio", "command": "tavily-cmd"}}})
    )
    sess = _session(_tool("tavily_search"))

    long_content = "x" * 500
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = f"Detailed Results:\n\nTitle: Test\nURL: https://example.com\nContent: {long_content}\nScore: 0.9\n"
    call_result = MagicMock()
    call_result.content = [text_block]
    sess.call_tool = AsyncMock(return_value=call_result)

    from familiar_agent.mcp_client import MCPClientManager

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, _patch_mcp([sess])
    ):
        mgr = MCPClientManager(config_path=cfg)
        await mgr.start()
        text, _ = await mgr.call("tavily_search", {"query": "test"})

    assert "Score:" not in text
    content_line = next((ln for ln in text.splitlines() if ln.startswith("Content: ")), "")
    assert content_line.endswith("…")


@pytest.mark.asyncio
async def test_tavily_search_normalizes_invalid_time_range(tmp_path: Path) -> None:
    """tavily_search: invalid time_range values like '24h' and '7d' are normalized."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"tavily": {"type": "stdio", "command": "tavily-cmd"}}})
    )

    from familiar_agent.mcp_client import MCPClientManager

    for invalid, expected in [("24h", "day"), ("7d", "week"), ("30d", "month"), ("48h", "day"), ("3d", "week")]:
        sess = _session(_tool("tavily_search"))
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Results."
        call_result = MagicMock()
        call_result.content = [text_block]
        sess.call_tool = AsyncMock(return_value=call_result)

        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            sys.modules, _patch_mcp([sess])
        ):
            mgr = MCPClientManager(config_path=cfg)
            await mgr.start()
            await mgr.call("tavily_search", {"query": "news", "time_range": invalid})

        _, kwargs = sess.call_tool.call_args
        sent = kwargs["arguments"]
        assert sent["time_range"] == expected, f"{invalid!r} should map to {expected!r}, got {sent['time_range']!r}"


@pytest.mark.asyncio
async def test_tavily_search_valid_time_range_unchanged(tmp_path: Path) -> None:
    """tavily_search: valid time_range values are passed through unchanged."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"tavily": {"type": "stdio", "command": "tavily-cmd"}}})
    )

    from familiar_agent.mcp_client import MCPClientManager

    for valid in ["day", "week", "month", "year", "d", "w", "m", "y"]:
        sess = _session(_tool("tavily_search"))
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Results."
        call_result = MagicMock()
        call_result.content = [text_block]
        sess.call_tool = AsyncMock(return_value=call_result)

        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            sys.modules, _patch_mcp([sess])
        ):
            mgr = MCPClientManager(config_path=cfg)
            await mgr.start()
            await mgr.call("tavily_search", {"query": "news", "time_range": valid})

        _, kwargs = sess.call_tool.call_args
        sent = kwargs["arguments"]
        assert sent["time_range"] == valid, f"Valid value {valid!r} should be unchanged"
