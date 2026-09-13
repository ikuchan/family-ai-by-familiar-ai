"""道具の失敗は「その道具はいま使えない」として運ぶ（出-o・2026-09-13）。

実機（J）：カレンダー MCP が `TypeError` を返し、それが「結果が届いた：TypeError…」と
**成功の形**で W に載った。主LLM は正しい引数で 4 回呼び直したが、同語の二度投げ禁止に
止められ、上限まで空回りして 25 秒後に「見つかりませんでした」と答えた。

直し：失敗は印（`Trigger.failed`・`Lookup.failed`）で運び、文は「道具が使えず失敗した」
だけにする（生の `TypeError` は対処できる情報ではない・ログにだけ残す）。失敗した道具は
その求めのあいだ候補から外す（構造で呼び直しを起こさない）。再試行はしない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.dif import DIF
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger
from tests.test_event_loop import _agent


def _ip_with_failing_calendar():
    a = _agent(stream_returns=[])
    mcp = MagicMock()
    mcp.call_result = AsyncMock(
        return_value=MagicMock(text="TypeError: unexpected keyword 'query'", image=None, ok=False)
    )
    ip = InformationProcessing(a)
    ip._dif = DIF(mcp=mcp)
    return a, ip


def test_a_failed_completion_carries_the_flag_and_not_the_raw_error():
    a, ip = _ip_with_failing_calendar()

    async def scenario():
        await ip._run_lookup("family_schedule", {}, "家族の予定を見る", None, 1)
        item = ip._triggers.get_nowait()
        await ip.close()
        return item

    item = asyncio.run(scenario())
    assert item.kind == "完了" and item.failed is True
    assert "道具が使えず失敗した" in item.result
    assert "TypeError" not in item.result


def test_a_successful_completion_is_not_flagged():
    a = _agent(stream_returns=[])
    mcp = MagicMock()
    mcp.call_result = AsyncMock(
        return_value=MagicMock(text="予定は入っていない。", image=None, ok=True)
    )
    ip = InformationProcessing(a)
    ip._dif = DIF(mcp=mcp)

    async def scenario():
        await ip._run_lookup("family_schedule", {}, "家族の予定を見る", None, 1)
        item = ip._triggers.get_nowait()
        await ip.close()
        return item

    item = asyncio.run(scenario())
    assert item.failed is False and "予定は入っていない" in item.result


def test_push_completion_accepts_the_flag():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.lookups.append(
        Lookup(index=1, action="search_deferred", query="今日の天気", generation=0)
    )
    ip.push_completion("今日の天気", "検索の道具が使えなかった", failed=True)
    item = ip._triggers.get_nowait()
    assert isinstance(item, Trigger) and item.failed is True


def test_the_version_says_the_tool_failed_not_that_a_result_arrived():
    """版（W に載る求めの状態）は「結果が届いた：…」でなく「道具が使えず失敗した」と書く。"""
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.request_text = "今日の予定は？"
    lk = Lookup(index=1, action="family_schedule", query="家族の予定を見る", generation=0)
    lk.result = "「家族の予定を見る」は道具が使えず失敗した"
    lk.failed = True
    ip._req.lookups.append(lk)
    text = ip._version_content()
    assert "1番：family_schedule「家族の予定を見る」は道具が使えず失敗した" in text
    assert "結果が届いた" not in text


# ── 失敗した道具は、その求めのあいだ候補から外す ─────────────────────────────


def test_a_failed_tool_leaves_the_main_llms_tool_list_for_the_rest_of_the_request():
    from tests.test_house_rules_tool import _agent_with_mcp, _tool_names

    a = _agent_with_mcp(["get_family_schedule", "get_house_rules"])
    ip = InformationProcessing(a)
    assert "get_family_schedule" in _tool_names(a, ("family_schedule", "house_rules"))
    ip._req.failed_actions.add("family_schedule")
    got = {d["name"] for d in ip._tools(actions=("say", "family_schedule", "house_rules"))}
    assert "get_family_schedule" not in got and "get_house_rules" in got


def test_a_failure_marks_the_action_family_whichever_name_called_it():
    """主LLM は道具名（`get_family_schedule`）で呼び、調停は動作名（`family_schedule`）で選ぶ。
    どちらの名で失敗しても、両方の入口から消える。"""
    from familiar_agent.loop.event_loop import _action_family

    assert _action_family("get_family_schedule") == "family_schedule"
    assert _action_family("family_schedule") == "family_schedule"
    assert _action_family("get_house_rules") == "house_rules"
    assert _action_family("search_notion") == "notion_search"
    assert _action_family("get_journal") == "journal"
    assert _action_family("search_deferred") == "search_deferred"


def test_intake_of_a_failed_completion_records_the_failed_action():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.lookups.append(
        Lookup(index=1, action="get_family_schedule", query="家族の予定を見る", generation=0)
    )
    ip._triggers.put_nowait(
        Trigger(
            kind="完了",
            query="家族の予定を見る",
            result="「家族の予定を見る」は道具が使えず失敗した",
            index=1,
            failed=True,
        )
    )

    async def scenario():
        await ip._intake()
        await ip.close()

    asyncio.run(scenario())
    assert ip._req.lookups[0].failed is True
    assert "family_schedule" in ip._req.failed_actions


def test_the_arbiter_is_not_offered_a_failed_tool():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._ACTIONS = {
        **ip._ACTIONS,
        "family_schedule": lambda ip: [{"name": "get_family_schedule"}],
        "house_rules": lambda ip: [{"name": "get_house_rules"}],
    }
    ip._req.failed_actions.add("family_schedule")
    assert ip._extra_actions() == ("house_rules",)


def test_the_failed_set_is_cleared_when_the_request_closes():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.failed_actions.add("family_schedule")
    asyncio.run(ip._abort_lookups())
    assert ip._req.failed_actions == set()
