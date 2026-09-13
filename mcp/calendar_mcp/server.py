"""家の予定を MCP で引く stdio サーバー（知-j・依存ゼロ）。

`obsidian-memo` の `memo_mcp/server.py` と同じ作法：SDK を使わず、`initialize`／`ping`／
`tools/list`／`tools/call` だけを持つ。**stdout は JSON-RPC 専用**（ログは stderr）。道具の
失敗は protocol error でなく result の `isError` に載せる（モデルが失敗を見られるように）。

取り方（2026-09-13 決定）：ファミリーカレンダーの**秘密の ICS**（`FAMILY_CALENDAR_ICS`）を
HTTPS で直読する。認証コード無し・読み取り専用。更新の遅れは実測してから判断する。
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from . import __version__
from .ics import TZ, events_between, parse_ics

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
_WEEKDAY_JA = ("月", "火", "水", "木", "金", "土", "日")

DESCRIPTION = (
    "家族の予定（ファミリーカレンダー）を返す。今日から days 日ぶん（既定 1＝今日だけ）。"
    "「今日／明日の予定は」「何時から」「今日何かあった？」に引く。家族の誰と話しているときでも使ってよい。"
    "返りの先頭に【いま】（日付・曜日・時刻）と【出典】（取得時刻）が付く。"
    "⚠ 記憶（recall）とは別のもの。カレンダーに書かれた予定だけを返し、会話の記憶は含まない。"
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 14,
            "description": "今日から何日ぶん（既定 1）",
        }
    },
    "additionalProperties": False,
}


def fetch_ics_http(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "familiar-ai calendar_mcp"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


@dataclass
class Server:
    ics_url: str
    fetch_ics: Callable[[str], str] = fetch_ics_http
    now: Callable[[], datetime] = lambda: datetime.now(TZ)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tools["get_family_schedule"] = {
            "name": "get_family_schedule",
            "description": DESCRIPTION,
            "inputSchema": INPUT_SCHEMA,
        }

    # ---- 道具の中身 -------------------------------------------------------

    def get_family_schedule(self, days: int = 1) -> str:
        if not self.ics_url:
            raise RuntimeError(
                "FAMILY_CALENDAR_ICS が設定されていない（ファミリーカレンダーの秘密の ICS の URL）"
            )
        days = max(1, min(14, int(days)))
        cal = parse_ics(self.fetch_ics(self.ics_url))
        now = self.now().astimezone(TZ)
        first = now.date()
        events = events_between(cal, first, days)
        head = (
            f"【いま】{now:%Y-%m-%d} {_WEEKDAY_JA[now.weekday()]}曜日 {now:%H:%M}（JST）\n"
            f"【出典】{cal.name or 'ファミリーカレンダー'}（{now:%H:%M} に取得・今日から {days} 日ぶん）\n"
        )
        if not events:
            return head + "\n予定は入っていない。"
        lines = []
        for e in events:
            day = f"{e.start_local:%m-%d}（{_WEEKDAY_JA[e.start_local.weekday()]}）"
            if e.all_day:
                when = "終日"
                body = f"{e.summary}（終日）"
            else:
                when = f"{e.start_local:%H:%M}" + (f"〜{e.end_local:%H:%M}" if e.end_local else "")
                body = f"{when} {e.summary}"
            note = f"　※{e.note}" if e.note else ""
            lines.append(f"- {day} {body}{note}")
        return head + "\n" + "\n".join(lines)

    # ---- JSON-RPC ---------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if msg_id is None:
            return None  # 通知
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                result: Any = {
                    "protocolVersion": asked
                    if asked in SUPPORTED_PROTOCOLS
                    else SUPPORTED_PROTOCOLS[0],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "family-calendar", "version": __version__},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": list(self.tools.values())}
            elif method == "tools/call":
                result = self._call(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"未対応のメソッド: {method}"},
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name != "get_family_schedule":
            return {
                "content": [{"type": "text", "text": f"そのツールはありません: {name}"}],
                "isError": True,
            }
        args = params.get("arguments") or {}
        try:
            text = self.get_family_schedule(**args)
        except Exception as exc:  # noqa: BLE001
            # 道具の失敗は result に載せる。protocol error にするとモデルが失敗を見られない。
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": text}]}

    def run(self) -> None:
        """stdio で回す。1 行 1 メッセージ（Content-Length 無しの newline 区切り）。"""
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            response = self.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def build(
    *, ics_url: str | None = None, fetch_ics: Callable[[str], str] | None = None, now=None
) -> Server:
    import os

    url = os.environ.get("FAMILY_CALENDAR_ICS", "") if ics_url is None else ics_url
    kwargs: dict[str, Any] = {"ics_url": url}
    if fetch_ics is not None:
        kwargs["fetch_ics"] = fetch_ics
    if now is not None:
        kwargs["now"] = now
    return Server(**kwargs)
