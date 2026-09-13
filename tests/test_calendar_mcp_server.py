"""カレンダー MCP サーバー（stdio・依存ゼロ）の契約（知-j）。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))

from calendar_mcp import server as srv  # noqa: E402
from calendar_mcp.ics import TZ  # noqa: E402

_ICS = """BEGIN:VCALENDAR
X-WR-CALNAME:ファミリー
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260914
DTEND;VALUE=DATE:20260915
SUMMARY:運動会
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Tokyo:20260914T153000
DTEND;TZID=Asia/Tokyo:20260914T170000
SUMMARY:テニス
END:VEVENT
END:VCALENDAR
"""


def _server(fetch=lambda url: _ICS, url="https://example/basic.ics"):
    return srv.build(
        fetch_ics=fetch, ics_url=url, now=lambda: datetime(2026, 9, 14, 8, 0, tzinfo=TZ)
    )


def test_tools_list_exposes_the_family_schedule() -> None:
    s = _server()
    res = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in res["result"]["tools"]]
    assert names == ["get_family_schedule"]
    tool = res["result"]["tools"][0]
    assert "days" in tool["inputSchema"]["properties"]
    assert "家族" in tool["description"]


def test_calling_returns_today_with_now_and_source() -> None:
    s = _server()
    res = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_family_schedule", "arguments": {}},
        }
    )
    text = res["result"]["content"][0]["text"]
    assert text.startswith("【いま】2026-09-14 月曜日 08:00")
    assert "【出典】ファミリー" in text
    assert "運動会（終日）" in text and "15:30〜17:00 テニス" in text
    assert not res["result"].get("isError")


def test_no_events_says_so() -> None:
    s = _server(fetch=lambda url: "BEGIN:VCALENDAR\nX-WR-CALNAME:ファミリー\nEND:VCALENDAR\n")
    res = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_family_schedule", "arguments": {"days": 2}},
        }
    )
    assert "予定は入っていない" in res["result"]["content"][0]["text"]


def test_missing_url_is_an_error_in_the_result_not_a_crash() -> None:
    s = _server(url="")
    res = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_family_schedule", "arguments": {}},
        }
    )
    assert res["result"]["isError"] is True
    assert "FAMILY_CALENDAR_ICS" in res["result"]["content"][0]["text"]


def test_a_failed_fetch_is_an_error_in_the_result() -> None:
    def boom(url):
        raise OSError("timed out")

    s = _server(fetch=boom)
    res = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "get_family_schedule", "arguments": {}},
        }
    )
    assert res["result"]["isError"] is True and "timed out" in res["result"]["content"][0]["text"]


def test_initialize_and_ping() -> None:
    s = _server()
    init = s.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
    )
    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert s.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
