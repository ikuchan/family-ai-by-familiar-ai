"""Deferred search tool: fire-and-forget search, result injected on next turn."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..core.structured_ask import ask_yes_no

if TYPE_CHECKING:
    from ..core.search_relevance import Checked

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 3
_MAX_PENDING = 10
_SEARCH_TIMEOUT_SEC = 30


async def _cancel_after(task: "asyncio.Task[object]", delay: float) -> None:
    await asyncio.sleep(delay)
    if not task.done():
        task.cancel()


_SOURCE_TO_TOOL = {
    "brave": "brave_web_search",
    "tavily": "tavily_search",
}

_SAME_INTENT_PROMPT = (
    "次の2つの検索クエリは同じ調査の意図ですか？ yes か no だけ答えてください。\nA: {a}\nB: {b}"
)


class DeferredSearchTool:
    """Starts an MCP search in the background and returns immediately.

    The caller gets an instant acknowledgement; completed results are
    injected into the system-prompt variable block on the next turn via
    ``pending_context()``.
    """

    def __init__(
        self,
        search_fn: Callable[[str, dict], Awaitable[Any]],  # MCP の `call_result`（本文・画像・ok）
        utility_backend: Any = None,
        context: Any = None,
        judge: "Callable[[str, str, str], Awaitable[Checked | None]] | None" = None,
    ) -> None:
        self._search_fn = search_fn
        # 返りが問いに合うかを照らす口（知-an）。(問い, 返りの文, 返した道具) → 合う結果だけの文と読みに行く
        # ページ（知-bm）、1 件も合わなければ None。渡さなければ照らさない。
        self._judge = judge
        self._utility_backend = utility_backend
        # `context(stance)` は立ち位置と文脈を返す（出-e）。渡さなければ
        # 立ち位置を渡さない（いままでと同じ）。
        self._context = context
        self._pending: list[dict] = []
        # 完了の渡し先（RH → 完了キュー）。繋がっていれば溜めずにここへ渡す。
        # 繋がっていなければ従来どおり `_pending` に溜めてポーリングで拾われる（排他）。
        self._completion_sink = None
        # 投げた語ごとの意図の uuid（出-cc）。完了と一緒に返し、ループは uuid で器を引く。
        self._intent_of: dict[str, str] = {}
        self._running: int = 0
        self._running_queries: set[str] = set()
        self._user_turn: bool = False

    def set_completion_sink(self, sink) -> None:
        """完了の渡し先を繋ぐ（引数は (query, result, *, failed)）。"""
        self._completion_sink = sink

    def _deliver(self, query: str, result: str, *, failed: bool = False) -> bool:
        """完了を渡し先へ。渡せたら True（溜めない＝二重配信を避ける）。

        `failed` は**道具が使えなかった**（出-o）。結果ではないので、受け手は候補から外す。
        """
        if self._completion_sink is None:
            return False
        try:
            intent_id = self._intent_of.pop(query, None)
            if intent_id is None:
                self._completion_sink(query, result, failed=failed)
            else:  # 意図の uuid を持つときだけ渡す（出-cc）
                self._completion_sink(query, result, failed=failed, intent_id=intent_id)
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
                    # 担い手の使い分け（知-am・2026-10-07）。書いていなかったので、数字が要る問い（天気）でも Brave を
                    # 使い「リンクの一覧しか出てこない」と答えた。
                    # 数字が要るかでは分けられなかった（知-bm・2026-10-22）。変わるか変わらないかで分ける。
                    "担い手は問いで選ぶ：いまの情報（天気・ニュース・値段・時刻表・試合の結果など、日によって変わること）は "
                    "Tavily。変わらない事実（意味・由来・歴史・人や物の説明など）は Brave。Brave の結果に答えが無ければ、"
                    "合いそうなページを機械が読んで一緒に届ける。"
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
                            "description": "担い手（省略時: brave）。日によって変わることなら tavily、変わらない事実なら brave",
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
        # **読めなかったら文字列の一致で決める**（出-d）。以前は `.startswith("yes")` で
        # 見ており、モデルが「はい、同じです」と日本語で答えると**偽**になった。偽になると
        # 同じ調査を二重に投げる（`求めの版チェーン`「同じ語は二度と調べない」）。
        # 例外時の落とし先とも揃える（以前は例外なら文字列比較、読めなければ「いいえ」と
        # 割れていた）。
        # **外から測る立ち位置。** 語の比較で、感情も人格も関わらない。
        from ..core.context_parts import Stance

        # `getattr` で引くのは、`__init__` を通さずに組み立てる呼び出しがあるためである
        # （`getattr` 形は grep に出ないので、撤去のときは名前で探しても見つからない）。
        ctx = getattr(self, "_context", None)
        system = ctx(Stance.INSTRUMENT) if ctx else None
        answer = await ask_yes_no(
            self._utility_backend,
            _SAME_INTENT_PROMPT.format(a=existing_query, b=new_query),
            max_tokens=5,
            system=system,
        )
        return answer if answer is not None else (new_query == existing_query)

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, None]:
        if tool_name != "search_deferred":
            return f"Unknown tool: {tool_name}", None
        text, _dispatched = await self.dispatch(tool_input)
        return text, None

    async def dispatch(self, tool_input: dict) -> tuple[str, bool]:
        """検索を投げる。2つ目の返り値は**背景タスクを作ったか**である。

        作らずに帰る経路が3つある（クエリが空・同時実行の上限・同じ意図が進行中）。その
        場合は完了も時間切れも永久に来ないので、飛行中として数えた側が自分で閉じなければ
        ならない。文面で見分けると文言を直すたびに壊れるため、投げたかどうかを明示して返す。
        """
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "クエリが空です。", False

        if self._running >= _MAX_CONCURRENT:
            return (
                f"同時に検索できるのは {_MAX_CONCURRENT} 件までです。"
                "しばらくしてから再度お試しください。",
                False,
            )

        # Deduplicate: skip if same intent is already running or pending.
        # existing_queries is empty on the first search → no utility LLM call needed.
        existing_queries = list(self._running_queries) + [item["query"] for item in self._pending]
        for existing in existing_queries:
            if await self._is_same_intent(query, existing):
                return (
                    f"「{existing}」の調査がすでに進行中です。結果は次のターンで届きます。",
                    False,
                )

        source = str(tool_input.get("source", "brave"))
        mcp_tool = _SOURCE_TO_TOOL.get(source, "brave_web_search")
        user_initiated = self._user_turn
        # Increment synchronously before task starts to prevent race conditions.
        self._running += 1
        self._running_queries.add(query)
        if tool_input.get("_intent_id"):
            self._intent_of[query] = str(tool_input["_intent_id"])  # 検索の相手には渡さない
        task = asyncio.create_task(self._run(query, mcp_tool, source, user_initiated))
        asyncio.create_task(_cancel_after(task, _SEARCH_TIMEOUT_SEC))
        return (
            f"「{query}」を {source} でバックグラウンド検索中… 次のターンで結果をお知らせします。",
            True,
        )

    async def _read_page(self, url: str) -> str:
        """抜粋に答えが無かったときに読むページ（知-bm）。一覧に継ぐ文。読めなければ空（一覧だけ届ける）。"""
        try:
            r = await self._search_fn("fetch", {"url": url})  # 字数は fetch の既定（5,000 字）
        except Exception:  # noqa: BLE001
            logger.warning("deferred search ページを読めなかった（%s）", url, exc_info=True)
            return ""
        if not r.ok:
            logger.warning("deferred search ページを読めなかった（%s・%.120s）", url, r.text)
            return ""
        logger.info("deferred search ページを読んだ（%s・%d 字）", url, len(r.text))
        return f"\n\n[読んだページ] {url}\n{r.text}"

    async def _run(
        self, query: str, mcp_tool: str, source: str, user_initiated: bool = False
    ) -> None:
        logger.debug("deferred search _run started (query=%r mcp_tool=%r)", query, mcp_tool)
        try:
            # 検索の道具は 2 つ（Brave／Tavily）。指定された側が使えなければ、もう片方で
            # 続ける（出-o）。同じ道具を叩き直す再試行はしない——失敗の中身（鍵・上限・
            # 型）は対処できる情報でなく、代わりがあるならそれで続けるのが正しい。
            r = await self._search_fn(mcp_tool, {"query": query})
            if not r.ok:
                other = next((t for t in _SOURCE_TO_TOOL.values() if t != mcp_tool), None)
                logger.warning(
                    "deferred search %s が使えなかった（%.120s）→ %s", mcp_tool, r.text, other
                )
                if other is not None:
                    r = await self._search_fn(other, {"query": query})
                    mcp_tool = other
            if r.ok and self._judge is not None:
                # 3 択か 2 択かは、頼んだ担い手でなく実際に返した道具で決める（知-bm）
                checked = await self._judge(query, r.text, mcp_tool)
                if checked is None:
                    # 全部外れた（知-an）。もう一方の検索で 1 回だけ調べ直す。外れた返りは届けない——届けると
                    # ループの調べものが済みになり、つなぎも止まる。
                    other = next((t for t in _SOURCE_TO_TOOL.values() if t != mcp_tool), None)
                    logger.info(
                        "deferred search %s の返りが問いに合わない → %s で調べ直す", mcp_tool, other
                    )
                    r2 = await self._search_fn(other, {"query": query}) if other else None
                    checked = (
                        await self._judge(query, r2.text, other)
                        if other is not None and r2 is not None and r2.ok
                        else None
                    )
                if checked is None:
                    _msg = f"「{query}」は関係のある結果が見つからなかった"
                    if (
                        not self._deliver(query, _msg, failed=True)
                        and len(self._pending) < _MAX_PENDING
                    ):
                        self._pending.append(
                            {
                                "query": query,
                                "result": _msg,
                                "source": source,
                                "user_initiated": user_initiated,
                            }
                        )
                    return
                r_text = checked.text
                if checked.read:
                    r_text += await self._read_page(checked.read)
            else:
                r_text = r.text
            if not r.ok:
                logger.warning("deferred search 両方の道具が使えなかった（%.120s）", r.text)
                _msg = f"「{query}」は検索の道具が使えず失敗した"
                if (
                    not self._deliver(query, _msg, failed=True)
                    and len(self._pending) < _MAX_PENDING
                ):
                    self._pending.append(
                        {
                            "query": query,
                            "result": _msg,
                            "source": source,
                            "user_initiated": user_initiated,
                        }
                    )
                return
            result = r_text
            logger.debug(
                "deferred search _run completed (query=%r result_len=%d)", query, len(result)
            )
            if not self._deliver(query, result) and len(self._pending) < _MAX_PENDING:
                self._pending.append(
                    {
                        "query": query,
                        "result": result,
                        "source": source,
                        "user_initiated": user_initiated,
                    }
                )
        except asyncio.CancelledError:
            logger.warning(
                "deferred search timed out after %ds (query=%r)", _SEARCH_TIMEOUT_SEC, query
            )
            _msg = f"検索がタイムアウトしました（{_SEARCH_TIMEOUT_SEC}秒）: {query}"
            if not self._deliver(query, _msg, failed=True) and len(self._pending) < _MAX_PENDING:
                self._pending.append(
                    {
                        "query": query,
                        "result": _msg,
                        "source": source,
                        "user_initiated": user_initiated,
                    }
                )
            # **再送出する。** 締切の見張り（`_cancel_after`）だけでなく、終了時のキャンセルも
            # ここを通る。飲み込むと、止めようとしているのに止まらない節ができる。
            raise
        except Exception as exc:
            logger.warning("deferred search failed (query=%r): %s", query, exc)
            _msg = f"「{query}」は検索の道具が使えず失敗した"
            if not self._deliver(query, _msg, failed=True) and len(self._pending) < _MAX_PENDING:
                self._pending.append(
                    {
                        "query": query,
                        "result": _msg,
                        "source": source,
                        "user_initiated": user_initiated,
                    }
                )
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
