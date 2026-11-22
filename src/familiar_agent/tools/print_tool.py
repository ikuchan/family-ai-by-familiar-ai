"""印刷の道具（知-bo 段 2・2026-11-22・`設計方針_印刷`）。

主LLM だけが呼ぶ 1 本 `print_document`（調停の候補には載せない）。本人の決定（2026-11-21）：刷るのは直前の会話で出た中身で、
主LLM が題・日付・本文（箇条書き可）にまとめる。刷る前に題を言って 1 回確かめる（「『〇〇』を 1 枚刷るね。いい？」）。
確かめは機械が持つ（`confirm_state.PendingConfirm`・タイマーと同じ形）。「いい」なら機械が預かった入力で呼び直す
（`confirmed=True`・LLM の入力からは届かない）。刷ったことは O に「機器」として残す。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.completion_kind import NOT_PRINTED_YET
from ..io import printer as printer_io
from ..io.oif import MI
from ..person_memory_manager import AGENT_SELF_ID

if TYPE_CHECKING:
    from ..core.confirm_state import PendingConfirm

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "print_document",
        "description": (
            "家のプリンタで 1 枚刷る（白黒・片面・A4）。「印刷して」「プリントして」と頼まれたら、直前の会話で出た中身"
            "（調べものの答え・レシピ・予定の一覧など）を題・本文にまとめて渡す。本文は 1 行に 1 つ、箇条書きは行頭に「- 」。"
            "返りが「まだ刷っていない」なら、問いは機械が出す（もう一度呼ばない・「いい」なら confirm が刷る）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "題（例「カレーの作り方」）"},
                "date": {"type": "string", "description": "日付（無ければ今日）"},
                "body": {"type": "string", "description": "本文（箇条書きは行頭に「- 」）"},
            },
            "required": ["title", "body"],
        },
    },
]


class PrintTool:
    def __init__(
        self,
        *,
        oif: Any,
        speaker: Callable[[], str],
        printer: str,
        folder: Path,
        ask: "Callable[[PendingConfirm], Awaitable[None]] | None" = None,
        render: Callable[..., int] = printer_io.render_pdf,
        send: Callable[..., str] = printer_io.send_to_printer,
        now: "Callable[[], datetime] | None" = None,
    ) -> None:
        self._oif = oif
        self._speaker = speaker
        self._printer = printer
        self._folder = folder
        self._ask = ask  # 「確かめて」の預かりを置く口（`agent.ask_confirm`）
        self._render = render
        self._send = send
        self._now = now or (lambda: datetime.now().astimezone())

    def can_render(self) -> bool:
        """PDF を作れるか（Qt のアプリが動いているか）。作れなければ、道具を渡さない（家族に聞いてから断らない）。"""
        try:
            from PySide6.QtGui import QGuiApplication
        except ImportError:
            return False
        return QGuiApplication.instance() is not None

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in TOOL_DEFINITIONS]

    async def call(
        self, name: str, tool_input: dict, *, confirmed: bool = False
    ) -> tuple[str, bool]:
        """`confirmed` は機械だけが立てる（`agent.resolve_confirm`）。LLM の入力にあっても読まない。"""
        if name != "print_document":
            return f"そんな道具は無い：{name}", False
        title = str(tool_input.get("title") or "").strip()
        body = str(tool_input.get("body") or "").strip()
        if not title or not body:
            return "刷れない：題と本文が要る", False
        date_text = str(tool_input.get("date") or "").strip() or f"{self._now():%Y年%m月%d日}"
        if not confirmed:
            # 先に組んで枚数を数える（「1 枚刷るね」と言って 2 枚出さない）。ファイルは残さない。
            try:
                pages = self._pages(title, date_text, body)
            except (printer_io.PrintError, OSError, ImportError) as e:
                logger.warning("印刷の PDF を組めなかった：%s", e)
                return f"刷れない：{e}", False
            ask = f"『{title}』を {pages} 枚刷るね。いい？"
            if self._ask is not None:
                from ..core.confirm_state import PendingConfirm

                await self._ask(
                    PendingConfirm(
                        action="print_document",
                        tool_input={k: v for k, v in tool_input.items() if k != "confirmed"},
                        asked_at=time.time(),
                        text=ask,
                        what=f"『{title}』",
                    )
                )
            return f"{NOT_PRINTED_YET}。確かめてから——本人に一度聞く：「{ask}」", True
        path = self._folder / "print.pdf"
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            pages = self._render(title, date_text, body, path)
            job = await asyncio.to_thread(self._send, path, printer=self._printer)
        except (printer_io.PrintError, OSError, ImportError) as e:
            logger.warning("印刷できなかった：%s", e)
            return f"刷れなかった：{e}", False
        finally:
            with contextlib.suppress(OSError):  # 会話の中身をファイルに残さない（保存は DB だけ）
                path.unlink(missing_ok=True)
        who = self._speaker() or "誰か"
        await self._write(
            f"{who}に頼まれて『{title}』を {pages} 枚刷った（受付 {job or '番号なし'}）"
        )
        return f"刷った：『{title}』{pages} 枚（受付 {job or '番号なし'}）", True

    def _pages(self, title: str, date_text: str, body: str) -> int:
        """組んだときの枚数（数えるだけで、ファイルは消す）。"""
        self._folder.mkdir(parents=True, exist_ok=True)
        path = self._folder / "count.pdf"
        try:
            return int(self._render(title, date_text, body, path))
        finally:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    async def _write(self, content: str) -> "str | None":
        try:
            return await self._oif.write(
                MI(id="", content=content[:500], timestamp=None, direction="機器"),
                writer_id=AGENT_SELF_ID,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("印刷の記録を書けなかった: %s", e)
            return None
