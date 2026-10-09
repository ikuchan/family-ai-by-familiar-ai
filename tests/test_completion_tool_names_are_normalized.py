"""主LLM が道具の名前で呼んだ外部の道具も、完了を分ける前に動作の名前に揃える（出-ay 段 5b・2026-10-10）。

調停は動作の名前（`family_schedule`）で選ぶが、主LLM は道具の名前（`get_family_schedule`）で呼び、結果の記録にもその名前が
残る。完了の表は動作の名前で引くので、道具の名前のままだと表に当たらず古い問いに落ちていた（本人の決定ア：揃える）。
揃えるのは既にある `_action_family`（同じ表 `_MCP_LOOKUPS` から引く）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core.completion_kind import kind_of


@pytest.mark.parametrize(
    ("tool", "action"),
    [
        ("get_family_schedule", "family_schedule"),
        ("get_house_rules", "house_rules"),
        ("search_notion", "notion_search"),
        ("get_journal", "journal"),
    ],
)
def test_a_tool_name_reaches_the_arbiter_as_its_action(tool, action):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    inp = ip._arbiter_input(
        utterance="明日の予定は？",
        workspace_ctx="",
        present_ctx="",
        returned_lookups=((tool, False, "明日は 10 時から歯医者"),),
    )
    asyncio.run(ip.close())
    assert inp.returned == ((action, False, "明日は 10 時から歯医者"),)
    got = kind_of(action, failed=False, result="x", origin="発話")
    assert got is not None and got[0] == "頼まれた調べものの答えが届いた"
