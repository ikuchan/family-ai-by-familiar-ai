"""Deferred fetch tool: fire-and-forget URL fetch, result injected on next turn.

Used after search results identify a URL worth reading in depth.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 3
_MAX_PENDING = 10


class DeferredFetchTool:
    """Fetches a URL in the background and returns immediately.

    The caller gets an instant acknowledgement; completed page content is
    injected into the system-prompt variable block on the next turn via
    ``pending_context()``.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str, dict], Awaitable[tuple[str, Any]]],
    ) -> None:
        self._fetch_fn = fetch_fn
        self._pending: list[dict] = []
        self._running: int = 0

    # ── Tool definition ───────────────────────────────────────────────

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "fetch_deferred",
                "description": (
                    "バックグラウンドでURLを取得し、結果を待たずに即座に返答できるようにする。"
                    "検索結果で見つけたURLをさらに詳しく調べるときに使う。"
                    "「もっと詳しく調べてきます」と伝えて会話を続けたいときに使うこと。"
                    "結果は次のターンで自動的にコンテキストに提供される。"
                    "今すぐ結果が必要なときは fetch を使うこと。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "取得するURL",
                        },
                    },
                    "required": ["url"],
                },
            }
        ]

    # ── Tool execution ────────────────────────────────────────────────

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, None]:
        if tool_name != "fetch_deferred":
            return f"Unknown tool: {tool_name}", None

        url = str(tool_input.get("url", "")).strip()
        if not url:
            return "URLが空です。", None

        if self._running >= _MAX_CONCURRENT:
            return (
                f"同時に取得できるのは {_MAX_CONCURRENT} 件までです。"
                "しばらくしてから再度お試しください。",
                None,
            )

        self._running += 1  # increment synchronously before task starts to prevent race
        asyncio.create_task(self._run(url))
        return (
            f"「{url}」をバックグラウンドで取得中… 次のターンで内容をお知らせします。",
            None,
        )

    async def _run(self, url: str) -> None:
        try:
            result, _ = await self._fetch_fn("fetch", {"url": url})
            if len(self._pending) < _MAX_PENDING:
                self._pending.append({"url": url, "result": result})
        except Exception as exc:
            logger.warning("deferred fetch failed (url=%r): %s", url, exc)
            if len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "url": url,
                    "result": f"取得中にエラーが発生しました: {exc}",
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
            snippet = item["result"][:3000]
            parts.append(f"[バックグラウンド取得完了: {item['url']}]\n{snippet}")
        self._pending.clear()
        return "\n\n".join(parts)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def is_running(self) -> bool:
        return self._running > 0
