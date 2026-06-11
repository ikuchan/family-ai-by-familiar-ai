"""MCP (Model Context Protocol) client manager.

Connects to external MCP servers and exposes their tools to the agent.
Body-related tools (camera, TTS, mobility) stay as built-in; MCP is for everything else.

Supported transports
--------------------
* **stdio** — launch a local subprocess (default)
* **sse** — connect to an HTTP+SSE server

Config file: ~/.familiar-ai.json  (same mcpServers format as Claude Code's ~/.claude.json)
Override:    MCP_CONFIG=/path/to/config.json

Example config:
    {
      "mcpServers": {
        "filesystem": {
          "type": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
        },
        "memory": {
          "type": "sse",
          "url": "http://localhost:3000/sse"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path.home() / ".familiar-ai.json"


def _resolve_config_path() -> Path:
    env = os.environ.get("MCP_CONFIG", "")
    return Path(env) if env else _DEFAULT_CONFIG


def _load_servers(config_path: Path) -> dict[str, dict[str, Any]]:
    """Read mcpServers from the config file. Returns {} if file absent or malformed."""
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            logger.warning("MCP config: mcpServers must be an object, ignoring")
            return {}
        return servers
    except Exception as e:
        logger.warning("Failed to load MCP config %s: %s", config_path, e)
        return {}


_TAVILY_SKIP_PREFIXES = ("Score: ", "Favicon: ", "Raw Content: ")
_TAVILY_CONTENT_CHARS = 120


def _compress_tavily_result(text: str) -> str:
    """Strip low-value fields and truncate Content snippets in a Tavily result.

    Keeps: Answer (if present), Title, URL, Content (first 120 chars).
    Drops: Score, Favicon, Raw Content, image lines.
    Reduces typical output from ~1200 tokens to ~300 tokens.
    """
    out: list[str] = []
    for line in text.splitlines():
        if any(line.startswith(p) for p in _TAVILY_SKIP_PREFIXES):
            continue
        if line.startswith("Content: "):
            body = line[len("Content: "):]
            if len(body) > _TAVILY_CONTENT_CHARS:
                body = body[:_TAVILY_CONTENT_CHARS] + "…"
            out.append(f"Content: {body}")
        else:
            out.append(line)
    return "\n".join(out)


_CONNECT_TIMEOUT = float(os.environ.get("MCP_CONNECT_TIMEOUT", "30"))


class MCPClientManager:
    """Manages MCP server connections (stdio and SSE) for the duration of the agent session."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _resolve_config_path()
        self._servers = _load_servers(self._config_path)
        self._sessions: dict[str, Any] = {}  # server_name → ClientSession
        # tool_name → server_name (for routing)
        self._tool_router: dict[str, str] = {}
        # Cached tool definitions (Anthropic format)
        self._tool_defs: list[dict[str, Any]] = []
        self._exit_stack = AsyncExitStack()
        self._started = False  # True once start() has been called (re-entry guard)
        self._start_complete = False  # True once start() has finished (success or failure)
        self._failed_servers: list[str] = []

    @property
    def is_started(self) -> bool:
        """True once start() has been called.  Does NOT imply any server is connected."""
        return self._started

    @property
    def is_connected(self) -> bool:
        """True if at least one server connected and registered tools."""
        return bool(self._sessions)

    async def _register_tools(self, name: str, session: Any) -> int:
        """Register tools from a connected session. Returns count of registered tools."""
        tools_result = await session.list_tools()
        tools = tools_result.tools if hasattr(tools_result, "tools") else []
        count = 0
        for tool in tools:
            tool_name: str = tool.name
            if tool_name in self._tool_router:
                existing = self._tool_router[tool_name]
                logger.warning(
                    "MCP tool name collision: '%s' provided by both '%s' and '%s'; '%s' wins",
                    tool_name,
                    existing,
                    name,
                    existing,
                )
                continue

            self._tool_router[tool_name] = name
            self._tool_defs.append(
                {
                    "name": tool_name,
                    "description": tool.description or "",
                    "input_schema": (
                        tool.inputSchema
                        if isinstance(tool.inputSchema, dict)
                        else {"type": "object", "properties": {}}
                    ),
                }
            )
            count += 1
        return count

    async def _connect_one(
        self,
        name: str,
        cfg: dict[str, Any],
        ClientSession: Any,
        StdioServerParameters: Any,
        stdio_client: Any,
    ) -> None:
        """Connect a single server and register its tools. Raises on failure."""
        import asyncio
        import sniffio as _sniffio

        try:
            _lib = _sniffio.current_async_library()
        except Exception as _e:
            _lib = f"<detection failed: {_e!r}>"
        logger.info("MCP connect '%s': sniffio=%r task=%r", name, _lib, asyncio.current_task())

        server_type = cfg.get("type", "stdio")

        if server_type == "stdio":
            command = cfg.get("command", "")
            args: list[str] = cfg.get("args", [])
            env: dict[str, str] | None = cfg.get("env") or None

            if not command:
                logger.warning("MCP server '%s': missing 'command', skipping", name)
                return

            params = StdioServerParameters(command=command, args=args, env=env)
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session: Any = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=_CONNECT_TIMEOUT)

        elif server_type == "sse":
            from mcp.client.sse import sse_client as _sse_client

            url = cfg.get("url", "")
            if not url:
                logger.warning("MCP server '%s': missing 'url' for sse type, skipping", name)
                return

            read, write = await self._exit_stack.enter_async_context(_sse_client(url=url))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=_CONNECT_TIMEOUT)

        else:
            logger.warning("MCP server '%s': unsupported type '%s', skipping", name, server_type)
            return

        self._sessions[name] = session
        count = await self._register_tools(name, session)
        logger.info("Connected to MCP server '%s' (%d tools)", name, count)

    async def start(self) -> None:
        """Connect to all configured servers. Skips servers that fail to connect."""
        if self._started:
            return
        self._started = True

        if not self._servers:
            self._start_complete = True
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp package not installed; MCP support disabled")
            self._start_complete = True
            return

        await self._exit_stack.__aenter__()

        for name, cfg in self._servers.items():
            try:
                await self._connect_one(name, cfg, ClientSession, StdioServerParameters, stdio_client)
            except Exception as e:
                self._failed_servers.append(name)
                logger.warning(
                    "Failed to connect to MCP server '%s': %r",
                    name,
                    e,
                    exc_info=True,
                )

        if self._failed_servers:
            logger.warning(
                "MCP: %d/%d server(s) failed to connect: %s",
                len(self._failed_servers),
                len(self._servers),
                ", ".join(self._failed_servers),
            )
        self._start_complete = True

    async def stop(self) -> None:
        """Close all MCP connections."""
        if not self._started:
            return
        try:
            await self._exit_stack.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("MCP cleanup error: %r", e)

    async def reset(self) -> None:
        """Stop all connections and reset state so start() can be called again."""
        await self.stop()
        self._sessions.clear()
        self._tool_router.clear()
        self._tool_defs.clear()
        self._failed_servers.clear()
        self._exit_stack = AsyncExitStack()
        self._started = False
        self._start_complete = False

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Anthropic-format tool definitions from all connected servers."""
        return list(self._tool_defs)

    async def call(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[str, str | None]:
        """Call a tool on the appropriate MCP server. Never raises — returns error as text."""
        server_name = self._tool_router.get(tool_name)
        if server_name is None:
            return f"MCP tool '{tool_name}' not found.", None

        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server '{server_name}' is not connected.", None

        if tool_name == "tavily_search":
            tool_input = {k: v for k, v in tool_input.items() if k != "country"}
            _time_range_map = {"24h": "day", "7d": "week", "30d": "month", "1h": "day", "48h": "day", "3d": "week"}
            if "time_range" in tool_input and tool_input["time_range"] in _time_range_map:
                tool_input = dict(tool_input)
                tool_input["time_range"] = _time_range_map[tool_input["time_range"]]

        _call_timeout = float(os.environ.get("MCP_CALL_TIMEOUT", "30"))
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=tool_input),
                timeout=_call_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '%s' call timed out after %.0fs", tool_name, _call_timeout)
            return f"MCP tool '{tool_name}' timed out after {_call_timeout:.0f}s", None
        except Exception as e:
            logger.warning("MCP tool '%s' call failed: %s", tool_name, e)
            return f"MCP tool '{tool_name}' error: {e}", None

        # Extract text and optional image from content blocks
        text_parts: list[str] = []
        image_b64: str | None = None

        content = result.content if hasattr(result, "content") else []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(item.text)
            elif item_type == "image":
                # item.data is already base64, item.mimeType e.g. "image/jpeg"
                image_b64 = item.data

        text = "\n".join(text_parts) if text_parts else "(no output)"
        if tool_name == "tavily_search":
            text = _compress_tavily_result(text)
        return text, image_b64
