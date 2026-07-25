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
_FETCH_TIMEOUT_SEC = 60


async def _cancel_after(task: "asyncio.Task[object]", delay: float) -> None:
    await asyncio.sleep(delay)
    if not task.done():
        task.cancel()


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
        # 完了の渡し先（RH → 完了キュー）。繋がっていれば溜めずに渡す（二重配信を避ける）。
        self._completion_sink = None
        self._running: int = 0
        self._user_turn: bool = False

    def set_user_turn(self, value: bool) -> None:
        """Mark whether the current agent turn is user-initiated (vs. autonomous desire)."""
        self._user_turn = value

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

        user_initiated = self._user_turn
        self._running += 1  # increment synchronously before task starts to prevent race
        task = asyncio.create_task(self._run(url, user_initiated))
        asyncio.create_task(_cancel_after(task, _FETCH_TIMEOUT_SEC))
        return (
            f"「{url}」をバックグラウンドで取得中… 次のターンで内容をお知らせします。",
            None,
        )

    def set_completion_sink(self, sink) -> None:
        """完了の渡し先を繋ぐ（`EVENT_LOOP` on のとき・引数は (url, result)）。"""
        self._completion_sink = sink

    def _deliver(self, url: str, result: str) -> bool:
        if self._completion_sink is None:
            return False
        try:
            self._completion_sink(url, result)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("完了キューへ渡せなかったので溜める: %s", e)
            return False

    async def _run(self, url: str, user_initiated: bool = False) -> None:
        try:
            result, _ = await self._fetch_fn("fetch", {"url": url})
            if not self._deliver(url, result) and len(self._pending) < _MAX_PENDING:
                self._pending.append({"url": url, "result": result, "user_initiated": user_initiated})
        except asyncio.CancelledError:
            logger.warning("deferred fetch timed out after %ds (url=%r)", _FETCH_TIMEOUT_SEC, url)
            _msg = f"取得がタイムアウトしました（{_FETCH_TIMEOUT_SEC}秒）: {url}"
            if not self._deliver(url, _msg) and len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "url": url,
                    "result": _msg,
                    "user_initiated": user_initiated,
                })
        except Exception as exc:
            logger.warning("deferred fetch failed (url=%r): %s", url, exc)
            _msg = f"取得中にエラーが発生しました: {exc}"
            if not self._deliver(url, _msg) and len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "url": url,
                    "result": _msg,
                    "user_initiated": user_initiated,
                })
        finally:
            self._running -= 1

    # ── Context injection ─────────────────────────────────────────────

    def pending_summary(self) -> str:
        """Return a comma-joined list of pending URLs (does not clear pending)."""
        return "、".join(item["url"] for item in self._pending)

    def pending_context(self) -> str:
        """Return all completed results as a context block, then clear them.

        The URL label is intentionally omitted — it is embedded in the inner_voice
        directive instead, so the LLM never echoes it as output text.
        """
        if not self._pending:
            return ""
        parts = [item["result"][:3000] for item in self._pending]
        self._pending.clear()
        return "\n\n".join(parts)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def has_user_initiated_pending(self) -> bool:
        return any(item.get("user_initiated") for item in self._pending)

    @property
    def is_running(self) -> bool:
        return self._running > 0
