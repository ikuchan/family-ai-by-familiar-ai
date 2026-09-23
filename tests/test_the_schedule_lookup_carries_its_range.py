"""家族の予定は**期間つき**の求め（出-p・2026-09-14 実機）。

「明日の僕の予定わかる？」に、調停は `family_schedule` を選んだが入力が `{}`（今日だけ）で、
主LLM が `days=2` で呼び直した 4 回はすべて「すでに調べた語なので投げない：家族の予定を見る」
に止まり、上限まで空回りして「まだはっきり確認できない」と答えた。原因は 2 つ——
(1) 調停の経路は期間を渡せない、(2) 見出しが固定なので入力が違っても同じ求めに見える。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.dif import DIF
from familiar_agent.loop.arbiter import _parse
from familiar_agent.loop.event_loop import (
    InformationProcessing,
    _query_label,
    _tool_input_for,
)
from tests.test_event_loop import _agent


def test_the_label_carries_the_range_so_a_wider_call_is_a_different_lookup() -> None:
    assert _query_label("family_schedule", {"days": 2}) == "家族の予定を見る（2 日ぶん）"
    assert _query_label("get_family_schedule", {"days": 2}) == "家族の予定を見る（2 日ぶん）"
    assert _query_label("family_schedule", {}) == "家族の予定を見る（1 日ぶん）"  # 道具の既定と同じ
    # 期間を持たない道具の見出しは固定のまま。
    assert _query_label("house_rules", {}) == "家の決まりを見る"
    assert _query_label("journal", {}) == "日次記録を見る"


def test_the_arbiter_query_becomes_the_number_of_days() -> None:
    assert _tool_input_for("family_schedule", "2") == {"days": 2}
    assert _tool_input_for("family_schedule", "明日（2）") == {"days": 2}
    assert _tool_input_for("family_schedule", "99") == {"days": 14}  # 道具の上限に丸める
    assert _tool_input_for("family_schedule", "0") == {"days": 1}
    assert _tool_input_for("family_schedule", "来週") == {}  # 読めなければ道具の既定に任せる
    assert _tool_input_for("house_rules", "家の決まりを見る") == {}
    assert _tool_input_for("notion_search", "運動会") == {"query": "運動会"}


def test_the_arbiter_keeps_the_days_it_wrote_in_the_query() -> None:
    d = _parse(
        '{"branch": "action", "action": "family_schedule", "query": "2", "filler": "見てみますね"}',
        can_see=False,
        extra_actions=("family_schedule",),
    )
    assert d is not None and d.action == "family_schedule" and d.query == "2"
    # 書かなかったときも action は落とさない（`(c)` 分岐は query が空だと full へ落ちる）。
    d = _parse(
        '{"branch": "action", "action": "family_schedule", "filler": "見てみますね"}',
        can_see=False,
        extra_actions=("family_schedule",),
    )
    assert d is not None and d.action == "family_schedule" and d.query


def test_a_second_call_with_a_wider_range_is_dispatched_not_deduped() -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        mcp = MagicMock()
        mcp.call_result = AsyncMock(
            return_value=MagicMock(text="【いま】…\n予定は入っていない。", image=None, ok=True)
        )
        ip = InformationProcessing(a)
        ip._dif = DIF(mcp=mcp)
        ip._dispatch_lookup(
            "family_schedule", {"days": 1}, _query_label("family_schedule", {"days": 1}), None
        )
        ip._dispatch_lookup(
            "get_family_schedule",
            {"days": 2},
            _query_label("get_family_schedule", {"days": 2}),
            None,
        )
        await asyncio.sleep(0.05)
        calls = [c.args[1] for c in mcp.call_result.call_args_list]
        await ip.close()
        return calls, [lk.query for lk in ip._req.lookups]

    calls, queries = asyncio.run(scenario())
    assert calls == [{"days": 1}, {"days": 2}]
    assert queries == ["家族の予定を見る（1 日ぶん）", "家族の予定を見る（2 日ぶん）"]


def test_the_lookup_log_line_shows_the_tool_input(caplog) -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        mcp = MagicMock()
        mcp.call_result = AsyncMock(
            return_value=MagicMock(text="予定は入っていない。", image=None, ok=True)
        )
        ip = InformationProcessing(a)
        ip._dif = DIF(mcp=mcp)
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
            await ip._run_lookup(
                "family_schedule", {"days": 2}, "家族の予定を見る（2 日ぶん）", None, 1
            )
        await ip.close()

    asyncio.run(scenario())
    line = next(r.message for r in caplog.records if "調べもの" in r.message)
    assert "days=2" in line
