"""家族の予定は、調停からも主LLM からも 1 手で引ける（知-j）。

あわせて、MCP の道具（`get_house_rules`・`get_family_schedule`）が**実際には呼ばれていなかった**
穴を塞ぐ：`_run_lookup_body` は `search_deferred`／`fetch_deferred` 以外の動作を `_dif.lookup` へ
流しており、そこは検索と取得しか知らない。主LLM が `get_house_rules` を呼んでも、その名は
`_LOOKUP_ACTIONS` に無いので捨てられていた（実機で一度も呼ばれた記録が無い）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.dif import DIF
from familiar_agent.loop.arbiter import _parse, arbitrate
from familiar_agent.loop.event_loop import _LOOKUP_ACTIONS, InformationProcessing, _query_label

from tests.test_event_loop import _agent
from tests.test_house_rules_tool import _agent_with_mcp, _tool_names


def test_the_family_schedule_tool_is_offered_to_the_main_llm() -> None:
    got = _tool_names(_agent_with_mcp(["get_family_schedule"]), ("family_schedule",))
    assert got == {"get_family_schedule"}


def test_mcp_tool_names_count_as_lookups() -> None:
    assert "get_family_schedule" in _LOOKUP_ACTIONS and "get_house_rules" in _LOOKUP_ACTIONS


def test_the_query_labels_are_fixed_for_mcp_lookups() -> None:
    assert _query_label("family_schedule", {}) == "家族の予定を見る"
    assert _query_label("get_family_schedule", {"days": 2}) == "家族の予定を見る"
    assert _query_label("house_rules", {}) == "家の決まりを見る"
    assert _query_label("get_house_rules", {}) == "家の決まりを見る"


def test_running_the_lookup_calls_the_mcp_tool_and_queues_the_result() -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        mcp = MagicMock()
        mcp.call = AsyncMock(return_value=("【いま】…\n- 09-14（月） 運動会（終日）", None))
        ip = InformationProcessing(a)
        ip._dif = DIF(mcp=mcp)
        await ip._run_lookup("family_schedule", {"days": 1}, "家族の予定を見る", None, 1)
        item = ip._triggers.get_nowait()
        await ip.close()
        return mcp.call.call_args, item

    call, item = asyncio.run(scenario())
    assert call.args[0] == "get_family_schedule" and call.args[1] == {"days": 1}
    assert item.kind == "完了" and "運動会" in item.result


def test_the_arbiter_can_choose_the_family_schedule() -> None:
    d = _parse(
        '{"branch": "action", "action": "family_schedule", "text": "見てみますね"}',
        can_see=False,
        extra_actions=("family_schedule",),
    )
    assert d is not None and d.action == "family_schedule" and d.query == "家族の予定を見る"


def test_the_arbiter_prompt_offers_the_schedule_when_available() -> None:
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(
        arbitrate(
            b, utterance="今日の予定は？", workspace_ctx="", extra_actions=("family_schedule",)
        )
    )
    assert '"family_schedule"' in b.complete.call_args.args[0]
    b2 = MagicMock(spec=["complete"])
    b2.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(arbitrate(b2, utterance="今日の予定は？", workspace_ctx=""))
    assert '"family_schedule"' not in b2.complete.call_args.args[0]
