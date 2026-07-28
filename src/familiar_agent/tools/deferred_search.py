"""Deferred search tool: fire-and-forget search, result injected on next turn."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 3
_MAX_PENDING = 10
_SEARCH_TIMEOUT_SEC = 60


async def _cancel_after(task: "asyncio.Task[object]", delay: float) -> None:
    await asyncio.sleep(delay)
    if not task.done():
        task.cancel()
_SOURCE_TO_TOOL = {
    "brave": "brave_web_search",
    "tavily": "tavily_search",
}

_SAME_INTENT_PROMPT = (
    "次の2つの検索クエリは同じ調査の意図ですか？ yes か no だけ答えてください。\n"
    "A: {a}\nB: {b}"
)


class DeferredSearchTool:
    """Starts an MCP search in the background and returns immediately.

    The caller gets an instant acknowledgement; completed results are
    injected into the system-prompt variable block on the next turn via
    ``pending_context()``.
    """

    def __init__(
        self,
        search_fn: Callable[[str, dict], Awaitable[tuple[str, Any]]],
        utility_backend: Any = None,
    ) -> None:
        self._search_fn = search_fn
        self._utility_backend = utility_backend
        self._pending: list[dict] = []
        # 完了の渡し先（RH → 完了キュー）。繋がっていれば溜めずにここへ渡す。
        # 繋がっていなければ従来どおり `_pending` に溜めてポーリングで拾われる（排他）。
        self._completion_sink = None
        self._running: int = 0
        self._running_queries: set[str] = set()
        self._user_turn: bool = False

    def set_completion_sink(self, sink) -> None:
        """完了の渡し先を繋ぐ（引数は (query, result)）。"""
        self._completion_sink = sink

    def _deliver(self, query: str, result: str) -> bool:
        """完了を渡し先へ。渡せたら True（溜めない＝二重配信を避ける）。"""
        if self._completion_sink is None:
            return False
        try:
            self._completion_sink(query, result)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("完了キューへ渡せなかったので溜める: %s", e)
            return False

    def set_user_turn(self, value: bool) -> None:
        """Mark whether the current agent turn is user-initiated (vs. autonomous desire)."""
        self._user_turn = value

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
                    " 結果が届いたターンでは必ず say() を使って自分の言葉（だよ・みたい口語）で報告すること。"
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

    async def _is_same_intent(self, new_query: str, existing_query: str) -> bool:
        """Return True if new_query and existing_query share the same search intent."""
        if self._utility_backend is None:
            return new_query == existing_query
        try:
            answer = await self._utility_backend.complete(
                _SAME_INTENT_PROMPT.format(a=existing_query, b=new_query),
                max_tokens=5,
            )
            return answer.strip().lower().startswith("yes")
        except Exception:
            return new_query == existing_query

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, None]:
        if tool_name != "search_deferred":
            return f"Unknown tool: {tool_name}", None

        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "クエリが空です。", None

        if self._running >= _MAX_CONCURRENT:
            return (
                f"同時に検索できるのは {_MAX_CONCURRENT} 件までです。"
                "しばらくしてから再度お試しください。",
                None,
            )

        # Deduplicate: skip if same intent is already running or pending.
        # existing_queries is empty on the first search → no utility LLM call needed.
        existing_queries = list(self._running_queries) + [
            item["query"] for item in self._pending
        ]
        for existing in existing_queries:
            if await self._is_same_intent(query, existing):
                return (
                    f"「{existing}」の調査がすでに進行中です。結果は次のターンで届きます。",
                    None,
                )

        source = str(tool_input.get("source", "brave"))
        mcp_tool = _SOURCE_TO_TOOL.get(source, "brave_web_search")
        user_initiated = self._user_turn
        # Increment synchronously before task starts to prevent race conditions.
        self._running += 1
        self._running_queries.add(query)
        task = asyncio.create_task(self._run(query, mcp_tool, source, user_initiated))
        asyncio.create_task(_cancel_after(task, _SEARCH_TIMEOUT_SEC))
        return (
            f"「{query}」を {source} でバックグラウンド検索中… 次のターンで結果をお知らせします。",
            None,
        )

    async def _run(self, query: str, mcp_tool: str, source: str, user_initiated: bool = False) -> None:
        logger.debug("deferred search _run started (query=%r mcp_tool=%r)", query, mcp_tool)
        try:
            result, _ = await self._search_fn(mcp_tool, {"query": query})
            logger.debug("deferred search _run completed (query=%r result_len=%d)", query, len(result))
            if not self._deliver(query, result) and len(self._pending) < _MAX_PENDING:
                self._pending.append({"query": query, "result": result, "source": source, "user_initiated": user_initiated})
        except asyncio.CancelledError:
            logger.warning("deferred search timed out after %ds (query=%r)", _SEARCH_TIMEOUT_SEC, query)
            _msg = f"検索がタイムアウトしました（{_SEARCH_TIMEOUT_SEC}秒）: {query}"
            if not self._deliver(query, _msg) and len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "query": query,
                    "result": _msg,
                    "source": source,
                    "user_initiated": user_initiated,
                })
        except Exception as exc:
            logger.warning("deferred search failed (query=%r): %s", query, exc)
            _msg = f"検索中にエラーが発生しました: {exc}"
            if not self._deliver(query, _msg) and len(self._pending) < _MAX_PENDING:
                self._pending.append({
                    "query": query,
                    "result": _msg,
                    "source": source,
                    "user_initiated": user_initiated,
                })
        finally:
            self._running -= 1
            self._running_queries.discard(query)

    # ── Context injection ─────────────────────────────────────────────

    def pending_summary(self) -> str:
        """Return a comma-joined list of pending query strings (does not clear pending)."""
        return "、".join(item["query"] for item in self._pending)

    def pending_context(self) -> str:
        """Return all completed results as a context block, then clear them.

        The query label is intentionally omitted here — it is embedded in the
        inner_voice directive instead, so the LLM never echoes it as output text.
        """
        if not self._pending:
            return ""
        parts = [item["result"][:2000] for item in self._pending]
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
