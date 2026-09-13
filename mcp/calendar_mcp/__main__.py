"""`python -m calendar_mcp` で stdio サーバーが上がる。

    { "mcpServers": { "family-calendar": {
        "type": "stdio",
        "command": "/path/to/.venv/bin/python", "args": ["-m", "calendar_mcp"],
        "cwd": "/path/to/family-ai-by-familiar-ai/mcp",
        "env": { "FAMILY_CALENDAR_ICS": "https://calendar.google.com/calendar/ical/.../private-.../basic.ics" }
    }}}

`--selftest` で initialize → tools/list → tools/call を 1 往復ずつ流す（実際に ICS を取る）。
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
            "params": {"name": "get_family_schedule", "arguments": {"days": 2}},
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
