"""Notion を読む stdio MCP サーバー（知-k・依存ゼロ）。

`obsidian-memo` の `memo_mcp` と同じ作法（SDK 無し・stdout は JSON-RPC 専用・道具の失敗は
result の `isError`）。**中身は基本すべて家族ティア**（2026-09-13 の決定）なので、道具の名前に
人は入れない。鍵は `NOTION_TOKEN` か `~/.config/obsidian-memo.env`（memo_mcp と同じ置き場・
git に置かない）。権限の壁は「インテグレーションに共有した DB だけ読める」で作る。

この鍵から見える Notion（2026-09-13 実測）は 2 つのデータソース：**`ObsidianMemo 索引`**
（ページ名・abstract・パス・種別・確度・更新＝Vault の目次）と **`ObsidianMemo 日次記録`**
（対象日・朝の気分・疲労・睡眠）。索引のページに本文（ブロック）は無く、本文は Vault にある。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import __version__

TZ = ZoneInfo("Asia/Tokyo")
API = "https://api.notion.com/v1"
VERSION = "2026-03-11"  # memo_mcp と同じ
ENV_FILE = "~/.config/obsidian-memo.env"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
_WEEKDAY_JA = ("月", "火", "水", "木", "金", "土", "日")
_MIN_INTERVAL = 0.34  # 約 3 req/秒（公式の目安）

SEARCH_DESCRIPTION = (
    "家の記録の目次（Notion の ObsidianMemo 索引）を検索する。ページ名・要約（abstract）・"
    "記録の場所（パス）・種別・確度・更新日を返す。「前に決めたこと」「〜の経緯」「〜について書いてある？」"
    "に引く。家族の誰と話しているときでも使ってよい。返るのは目次であって本文ではない。"
    "⚠ 記憶（recall）とは別のもの。一緒に過ごした記憶でなく、書き溜めた記録の目次。"
)
JOURNAL_DESCRIPTION = (
    "日ごとの記録（Notion の ObsidianMemo 日次記録）を直近 days 日ぶん返す：朝の気分・肉体疲労・"
    "脳疲労・体感・睡眠。「最近よく眠れてる？」「調子どう？」に引く。家族の誰と話しているときでも使ってよい。"
)


def token_from_env() -> str:
    """**中身を出力しない。**あるか無いかだけ。"""
    if os.environ.get("NOTION_TOKEN"):
        return os.environ["NOTION_TOKEN"]
    path = pathlib.Path(ENV_FILE).expanduser()
    if not path.exists():
        return ""
    m = re.search(r"^NOTION_TOKEN=(\S+)", path.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else ""


def make_http_api(token: str) -> Callable[..., dict]:
    last = [0.0]

    def call(method: str, path: str, body: dict | None = None) -> dict:
        wait = _MIN_INTERVAL - (time.monotonic() - last[0])
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            API + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": VERSION,
                "Content-Type": "application/json",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                    last[0] = time.monotonic()
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                last[0] = time.monotonic()
                if e.code == 429:
                    time.sleep(float(e.headers.get("Retry-After", 2)))
                    continue
                if 500 <= e.code < 600 and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                # 本文に鍵は入らないが、中身をそのまま返さない
                raise RuntimeError(f"Notion API が {e.code} を返した（{method} {path}）") from None
        raise RuntimeError(f"Notion API が繰り返し失敗した（{method} {path}）")

    return call


# ---- プロパティの読み方 ------------------------------------------------------


def _plain(prop: dict | None) -> str:
    if not prop:
        return ""
    t = prop.get("type")
    if t in ("title", "rich_text"):
        return "".join(x.get("plain_text", "") for x in prop.get(t) or [])
    if t == "select":
        return str((prop.get("select") or {}).get("name", ""))
    if t == "date":
        return str((prop.get("date") or {}).get("start", ""))
    if t == "number":
        v = prop.get("number")
        return "" if v is None else (str(int(v)) if float(v).is_integer() else str(v))
    return ""


def _title(page: dict) -> str:
    for p in (page.get("properties") or {}).values():
        if p.get("type") == "title":
            return _plain(p)
    return ""


def _prop(page: dict, name: str) -> str:
    return _plain((page.get("properties") or {}).get(name))


@dataclass
class Server:
    api: Callable[..., dict]
    token: str
    journal_ds: str
    now: Callable[[], datetime] = lambda: datetime.now(TZ)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tools["search_notion"] = {
            "name": "search_notion",
            "description": SEARCH_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "探す語（日本語でよい）"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "件数（既定 5）",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
        self.tools["get_journal"] = {
            "name": "get_journal",
            "description": JOURNAL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "直近何日ぶん（既定 7）",
                    }
                },
                "additionalProperties": False,
            },
        }

    def _head(self, source: str) -> str:
        now = self.now().astimezone(TZ)
        return (
            f"【いま】{now:%Y-%m-%d} {_WEEKDAY_JA[now.weekday()]}曜日 {now:%H:%M}（JST）\n"
            f"【出典】{source}（{now:%H:%M} に取得）\n"
        )

    def search_notion(self, query: str, limit: int = 5) -> str:
        if not self.token:
            raise RuntimeError("NOTION_TOKEN が無い（`~/.config/obsidian-memo.env` か環境変数）")
        query = (query or "").strip()
        if not query:
            raise ValueError("query が空")
        limit = max(1, min(10, int(limit)))
        res = self.api(
            "POST",
            "/search",
            {"query": query, "page_size": limit, "filter": {"value": "page", "property": "object"}},
        )
        rows = [r for r in res.get("results") or [] if r.get("object") == "page"]
        head = self._head("Notion・ObsidianMemo 索引")
        if not rows:
            return head + f"\n「{query}」に当たる記録は見つからなかった。"
        lines = []
        for r in rows:
            title = _title(r) or "（無題）"
            abstract = _prop(r, "abstract")
            path = _prop(r, "パス")
            kind = _prop(r, "種別")
            conf = _prop(r, "確度")
            updated = _prop(r, "更新") or str(r.get("last_edited_time", ""))[:10]
            meta = "・".join(
                x
                for x in (
                    kind,
                    f"確度 {conf}" if conf else "",
                    f"更新 {updated}" if updated else "",
                )
                if x
            )
            line = f"- {title}（{meta}）"
            if abstract:
                line += f"：{abstract}"
            if path:
                line += f"　〔{path}〕"
            lines.append(line)
        return head + f"\n「{query}」で {len(rows)} 件：\n" + "\n".join(lines)

    def get_journal(self, days: int = 7) -> str:
        if not self.token:
            raise RuntimeError("NOTION_TOKEN が無い（`~/.config/obsidian-memo.env` か環境変数）")
        if not self.journal_ds:
            raise RuntimeError("NOTION_JOURNAL_DATA_SOURCE が無い（日次記録のデータソース id）")
        days = max(1, min(30, int(days)))
        since = (self.now().astimezone(TZ).date() - timedelta(days=days - 1)).isoformat()
        res = self.api(
            "POST",
            f"/data_sources/{self.journal_ds}/query",
            {
                # 日次記録は `日付`（date 型）を持つ（memo_mcp の SPECS と同じ列）。
                "filter": {"property": "日付", "date": {"on_or_after": since}},
                "sorts": [{"property": "日付", "direction": "descending"}],
                "page_size": days,
            },
        )
        rows = res.get("results") or []
        head = self._head("Notion・ObsidianMemo 日次記録")
        if not rows:
            return head + f"\n直近 {days} 日の記録は無い。"
        lines = []
        for r in rows:
            day = _title(r)
            parts = []
            for label, key, unit in (
                ("朝の気分", "朝の気分", ""),
                ("肉体疲労", "肉体疲労", ""),
                ("脳疲労", "脳疲労", ""),
                ("体感", "体感", ""),
                ("睡眠", "睡眠分", " 分"),
                ("睡眠スコア", "睡眠スコア", ""),
            ):
                v = _prop(r, key)
                if v:
                    parts.append(f"{label} {v}{unit}")
            lines.append(f"- {day}：" + "／".join(parts))
        return head + f"\n直近 {days} 日：\n" + "\n".join(lines)

    # ---- JSON-RPC ---------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if msg_id is None:
            return None
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                result: Any = {
                    "protocolVersion": asked
                    if asked in SUPPORTED_PROTOCOLS
                    else SUPPORTED_PROTOCOLS[0],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "notion-memo", "version": __version__},
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
        fn = {"search_notion": self.search_notion, "get_journal": self.get_journal}.get(name)
        if fn is None:
            return {
                "content": [{"type": "text", "text": f"そのツールはありません: {name}"}],
                "isError": True,
            }
        try:
            text = fn(**(params.get("arguments") or {}))
        except Exception as exc:  # noqa: BLE001
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": text}]}

    def run(self) -> None:
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


def _from_mcp_config(key: str) -> str:
    path = os.path.expanduser(os.environ.get("MCP_CONFIG", "~/.familiar-ai.json"))
    try:
        with open(path, encoding="utf-8") as fh:
            return str(json.load(fh)["mcpServers"]["notion-memo"]["env"][key])
    except Exception:  # noqa: BLE001
        return ""


def build(*, api=None, token: str | None = None, journal_ds: str | None = None, now=None) -> Server:
    tok = token_from_env() if token is None else token
    ds = (
        (
            os.environ.get("NOTION_JOURNAL_DATA_SOURCE")
            or _from_mcp_config("NOTION_JOURNAL_DATA_SOURCE")
        )
        if journal_ds is None
        else journal_ds
    )
    kwargs: dict[str, Any] = {"api": api or make_http_api(tok), "token": tok, "journal_ds": ds}
    if now is not None:
        kwargs["now"] = now
    return Server(**kwargs)
