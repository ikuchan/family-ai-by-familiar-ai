"""Deferred search tool: fire-and-forget search, result injected on next turn."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 3
_MAX_PENDING = 10
_SOURCE_TO_TOOL = {
    "brave": "brave_web_search",
    "tavily": "tavily_search",
}


class DeferredSearchTool:
    """Starts an MCP search in the background and returns immediately.

    The caller gets an instant acknowledgement; completed results are
    injected into the system-prompt variable block on the next turn via
    ``pending_context()``.
    """

    def __init__(
        self,
        search_fn: Callable[[str, dict], Awaitable[tuple[str, Any]]],
    ) -> None:
        self._search_fn = search_fn
        self._pending: list[dict] = []
        self._running: int = 0

    # ── Tool definition ───────────────────────────────────────────────

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "search_deferred",
                "description": (
                    "バックグラウンドで検索を開始し、結果を待たずに即座に返答できるようにする。"
                    "「調べておくね」と伝えて会話を続けたいときに使う。"
                    "結果は次のターンで自動的にコンテキストに提供される。"
                    "今すぐ結果が必要なときは brave_web_search / tavily_search を使うこと。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索クエリ",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["brave", "tavily"],
                            "description": "検索エンジン（省略時: brave）",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    # ── Tool execution ────────────────────────────────────────────────

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, None]:
        if tool_name != "search_deferred":
            return f"Unknown tool: {tool_name}", None

        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "クエリが空です。", None

        if self._running >= _MAX_CONCURRENT:
            return (
                f"バックグラウンド検索が混んでいます（同時 {_MAX_CONCURRENT} 件まで）。"
                "少し待ってから再度試してください。",
                None,
            )

        source = str(tool_input.get("source", "brave"))
        mcp_tool = _SOURCE_TO_TOOL.get(source, "brave_web_search")
        asyncio.create_task(self._run(query, mcp_tool, source))
        return (
            f"「{query}」を {source} でバックグラウンド検索中… 次のターンで結果をお知らせします。",
            None,
        )

    async def _run(self, query: str, mcp_tool: str, source: str) -> None:
        self._running += 1
        try:
            result, _ = await self._search_fn(mcp_tool, {"query": query})
            if len(self._pending) < _MAX_PENDING:
                self._pending.append({"query": query, "result": result, "source": source})
        except Exception as exc:
            logger.warning("deferred search failed (query=%r): %s", query, exc)
            if len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "query": query,
                    "result": f"検索中にエラーが発生しました: {exc}",
                    "source": source,
                })
        finally:
            self._running -= 1

    # ── Context injection ─────────────────────────────────────────────

    def pending_context(self) -> str:
        """Return all completed results as a context block, then clear them."""
        if not self._pending:
            return ""
        parts = []
        for item in self._pending:
            snippet = item["result"][:800]
            parts.append(f"[バックグラウンド検索完了: {item['query']}]\n{snippet}")
        self._pending.clear()
        return "\n\n".join(parts)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def is_running(self) -> bool:
        return self._running > 0
