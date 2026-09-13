"""`python -m notion_mcp` で stdio サーバーが上がる。

    { "mcpServers": { "notion-memo": {
        "type": "stdio",
        "command": "/path/to/.venv/bin/python", "args": ["-m", "notion_mcp"],
        "cwd": "/path/to/family-ai-by-familiar-ai/mcp",
        "env": { "NOTION_JOURNAL_DATA_SOURCE": "<日次記録のデータソース id>" }
    }}}

鍵は `~/.config/obsidian-memo.env` の `NOTION_TOKEN`（memo_mcp と同じ）。
`--selftest` で initialize → tools/list → search_notion を 1 往復ずつ流す（実際に Notion を読む）。
"""

from __future__ import annotations

import json
import sys

from .server import build


def selftest() -> int:
    server = build()
    ok = True
    for request in (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "selftest", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_notion", "arguments": {"query": "サッカー", "limit": 3}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_journal", "arguments": {"days": 3}},
        },
    ):
        response = server.handle(request)
        print(f"--> {request.get('method')}", file=sys.stderr)
        if response is None:
            continue
        if "error" in response or (response.get("result") or {}).get("isError"):
            ok = False
        print(json.dumps(response, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    build().run()
