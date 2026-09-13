"""Notion の目次と日次記録は、調停からも主LLM からも 1 手で引ける（知-k）。

`search_notion` は query が要る（`family_schedule`／`house_rules` は要らない）。見出しは
「Notion で『…』を探す」。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.dif import DIF
from familiar_agent.loop.arbiter import _parse, arbitrate
from familiar_agent.loop.event_loop import _LOOKUP_ACTIONS, InformationProcessing, _query_label

from tests.test_event_loop import _agent
from tests.test_house_rules_tool import _agent_with_mcp, _tool_names


def test_the_notion_tools_are_offered_to_the_main_llm() -> None:
    got = _tool_names(
        _agent_with_mcp(["search_notion", "get_journal"]), ("notion_search", "journal")
    )
    assert got == {"search_notion", "get_journal"}


def test_notion_tool_names_count_as_lookups() -> None:
    assert {"search_notion", "get_journal", "notion_search", "journal"} <= set(_LOOKUP_ACTIONS)


def test_the_query_label_carries_the_query_for_search() -> None:
    assert _query_label("notion_search", {"query": "サッカー"}) == "Notion で「サッカー」を探す"
    assert _query_label("search_notion", {"query": "サッカー"}) == "Notion で「サッカー」を探す"
    assert _query_label("journal", {"days": 3}) == "日次記録を見る"


def test_running_the_search_calls_the_mcp_tool_with_the_query() -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        mcp = MagicMock()
        mcp.call = AsyncMock(return_value=("「サッカー」で 3 件：…", None))
        ip = InformationProcessing(a)
        ip._dif = DIF(mcp=mcp)
        await ip._run_lookup(
            "notion_search", {"query": "サッカー"}, "Notion で「サッカー」を探す", None, 1
        )
        item = ip._triggers.get_nowait()
        await ip.close()
        return mcp.call.call_args, item

    call, item = asyncio.run(scenario())
    assert call.args[0] == "search_notion" and call.args[1] == {"query": "サッカー"}
    assert item.kind == "完了" and "3 件" in item.result


def test_the_arbiter_can_choose_notion_search_with_a_query() -> None:
    d = _parse(
        '{"branch": "action", "action": "notion_search", "query": "サッカー教室", "text": "見てみますね"}',
        extra_actions=("notion_search",),
    )
    assert d is not None and d.action == "notion_search" and d.query == "サッカー教室"


def test_the_arbiter_prompt_offers_notion_when_available() -> None:
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="サッカー教室の話、どこかに書いてある？",
            workspace_ctx="",
            extra_actions=("notion_search",),
        )
    )
    assert '"notion_search"' in b.complete.call_args.args[0]
