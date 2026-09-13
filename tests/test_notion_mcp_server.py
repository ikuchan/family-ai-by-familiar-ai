"""Notion を読む MCP サーバー（stdio・依存ゼロ・家族ティア）の契約（知-k）。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))

from notion_mcp import server as srv  # noqa: E402
from notion_mcp.server import TZ  # noqa: E402


def _rt(text):
    return [{"plain_text": text}]


_INDEX_HIT = {
    "object": "page",
    "id": "p1",
    "last_edited_time": "2026-08-30T10:00:00.000Z",
    "parent": {"type": "data_source_id", "database_id": "idx"},
    "properties": {
        "ページ": {"type": "title", "title": _rt("サッカー")},
        "abstract": {
            "type": "rich_text",
            "rich_text": _rt("2人の子が続けている活動。泰輝はレプロ、航輝はアレグレイル。"),
        },
        "パス": {"type": "rich_text", "rich_text": _rt("10_Wiki/テーマ/サッカー.md")},
        "種別": {"type": "select", "select": {"name": "topic"}},
        "確度": {"type": "select", "select": {"name": "medium"}},
        "更新": {"type": "date", "date": {"start": "2026-08-30"}},
    },
}
_JOURNAL_ROW = {
    "object": "page",
    "id": "j1",
    "properties": {
        "対象日": {"type": "title", "title": _rt("2026-09-12")},
        "朝の気分": {"type": "number", "number": 3.5},
        "肉体疲労": {"type": "number", "number": 2},
        "脳疲労": {"type": "number", "number": 4},
        "体感": {"type": "number", "number": 3},
        "睡眠分": {"type": "number", "number": 410},
        "睡眠スコア": {"type": "number", "number": 78},
    },
}


class _Api:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/search":
            return {"results": [_INDEX_HIT], "has_more": False}
        if path.startswith("/data_sources/") and path.endswith("/query"):
            return {"results": [_JOURNAL_ROW], "has_more": False}
        raise AssertionError(path)


def _server(api=None, token="ntn_x"):
    return srv.build(
        api=api or _Api(),
        token=token,
        journal_ds="ds-journal",
        now=lambda: datetime(2026, 9, 13, 15, 0, tzinfo=TZ),
    )


def test_tools_list_has_search_and_journal() -> None:
    res = _server().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in res["result"]["tools"]]
    assert names == ["search_notion", "get_journal"]
    assert all("_yusuke" not in n for n in names), "家族ティアなので名前に人は入らない"


def test_search_returns_title_abstract_path_and_dates() -> None:
    api = _Api()
    res = _server(api).handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search_notion", "arguments": {"query": "サッカー"}},
        }
    )
    text = res["result"]["content"][0]["text"]
    assert text.startswith("【いま】2026-09-13 日曜日 15:00")
    assert "サッカー" in text and "2人の子が続けている活動" in text
    assert "10_Wiki/テーマ/サッカー.md" in text and "更新 2026-08-30" in text and "topic" in text
    assert api.calls[0][2]["query"] == "サッカー"


def test_search_with_no_hits_says_so() -> None:
    class Empty(_Api):
        def __call__(self, method, path, body=None):
            return {"results": [], "has_more": False}

    res = _server(Empty()).handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_notion", "arguments": {"query": "宇宙"}},
        }
    )
    assert "見つからなかった" in res["result"]["content"][0]["text"]


def test_journal_returns_recent_days_as_lines() -> None:
    api = _Api()
    res = _server(api).handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_journal", "arguments": {"days": 7}},
        }
    )
    text = res["result"]["content"][0]["text"]
    assert "2026-09-12" in text and "朝の気分 3.5" in text and "睡眠 410 分" in text
    assert api.calls[0][1] == "/data_sources/ds-journal/query"


def test_missing_token_is_an_error_in_the_result() -> None:
    res = _server(token="").handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "search_notion", "arguments": {"query": "x"}},
        }
    )
    assert (
        res["result"]["isError"] is True and "NOTION_TOKEN" in res["result"]["content"][0]["text"]
    )


def test_journal_without_data_source_says_so() -> None:
    s = srv.build(
        api=_Api(),
        token="ntn_x",
        journal_ds="",
        now=lambda: datetime(2026, 9, 13, 15, 0, tzinfo=TZ),
    )
    res = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "get_journal", "arguments": {}},
        }
    )
    assert (
        res["result"]["isError"] is True
        and "NOTION_JOURNAL_DATA_SOURCE" in res["result"]["content"][0]["text"]
    )
