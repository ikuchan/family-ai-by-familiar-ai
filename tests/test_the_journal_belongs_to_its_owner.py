"""日次記録は持ち主と話しているときだけ引ける（知-az・2026-10-11・本人の決定）。

Notion の日次記録（朝の気分・疲労・睡眠）は 1 人（パパ）の体の記録なのに、道具の名前に人が入っておらず、
誰と話していても引けた（たいきの「よく眠れてる？」にパパの睡眠を答えうる）。持ち主の英字を道具の名前に
付け（`get_journal_<latin>`・`NOTION_JOURNAL_OWNER`）、vault と同じ話者ゲートに載せる。持ち主の設定が
無ければ `get_journal` のまま家族の誰とでも引ける。
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

from familiar_agent.core.voice_rules import careful_voice
from familiar_agent.loop.event_loop import _action_family, _query_label
from tests.test_vault_lookup import _ip

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))

from notion_mcp import server as srv  # noqa: E402
from tests.test_notion_mcp_server import _Api  # noqa: E402


def _server(owner: str):
    return srv.build(
        api=_Api(),
        token="ntn_x",
        journal_ds="ds-journal",
        journal_owner=owner,
        now=lambda: datetime(2026, 9, 13, 15, 0, tzinfo=srv.TZ),
    )


def _call(server, name: str) -> dict:
    return server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": {}},
        }
    )["result"]


# --- サーバーが名前に持ち主を付ける ---


def test_the_owner_is_in_the_tool_name_and_it_answers_by_that_name():
    server = _server("yusuke")
    res = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tool = next(t for t in res["result"]["tools"] if t["name"].startswith("get_journal"))
    assert tool["name"] == "get_journal_yusuke"
    assert "家族の誰と話しているときでも" not in tool["description"]
    assert not _call(server, "get_journal_yusuke").get("isError")
    assert _call(server, "get_journal").get("isError")


def test_without_an_owner_the_name_stays_for_the_whole_family():
    server = _server("")
    names = [
        t["name"]
        for t in server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"][
            "tools"
        ]
    ]
    assert "get_journal" in names
    assert not _call(server, "get_journal").get("isError")


# --- 話者ゲートに載る ---


def test_the_journal_is_offered_only_to_its_owner():
    ip = _ip("ゆうすけ", ["get_journal_yusuke", "ask_vault_yusuke"])
    assert [d["name"] for d in ip._tools(actions=("journal",))] == ["get_journal_yusuke"]
    assert "journal" in ip._extra_actions()
    assert "vault" in ip._extra_actions()  # 反証の側：vault はいまどおり
    for other in ("たいき", ""):
        ip = _ip(other, ["get_journal_yusuke", "ask_vault_yusuke"])
        assert ip._tools(actions=("journal",)) == []
        assert "journal" not in ip._extra_actions()
        assert "vault" not in ip._extra_actions()


def test_without_an_owner_anyone_gets_the_journal():
    ip = _ip("たいき", ["get_journal"])
    assert [d["name"] for d in ip._tools(actions=("journal",))] == ["get_journal"]
    assert "journal" in ip._extra_actions()


def test_the_journal_action_calls_the_owners_tool():
    ip = _ip("ゆうすけ", ["get_journal_yusuke"])
    assert ip._personal_tool_name("journal") == "get_journal_yusuke"
    assert ip._personal_tool_name("vault") == ""
    ip._dif.call_tool = AsyncMock(return_value=("よく眠れている", True))
    ip._req.failed_actions = set()

    async def run():
        await ip._run_lookup_body("journal", {}, "日次記録を見る", None)

    asyncio.run(run())
    assert ip._dif.call_tool.await_args.args[0] == "get_journal_yusuke"


def test_the_owners_tool_name_reads_as_the_journal():
    assert _action_family("get_journal_yusuke") == "journal"
    assert _action_family("get_journal") == "journal"
    assert _query_label("get_journal_yusuke", {}) == "日次記録を見る"
    assert careful_voice("full", ("get_journal_yusuke",))
