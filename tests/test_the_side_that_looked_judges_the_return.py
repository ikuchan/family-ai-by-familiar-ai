"""see の帰りは、**出した側**が判断する（`イベント駆動ループ` v0.44）。

主LLM が見ると決めたなら、その続きは主LLM（調停を飛ばす・写真つき）。調停が見ると決めた
なら、帰りも調停が判断する（`full` を選べば主LLM は写真つき、`light` なら即席のラベルで答える）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.loop.arbiter import Decision as ArbiterDecision
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    return a, InformationProcessing(a)


async def _returned_see(ip):
    ip._req.lookups.append(Lookup(index=1, action="see", query="目の前を見る", generation=0))
    ip._triggers.put_nowait(
        Trigger(kind="完了", query="目の前を見る", result="（見えた）", index=1)
    )
    await ip._intake()


def test_a_see_thrown_by_the_arbiter_comes_back_to_the_arbiter() -> None:
    async def scenario():
        a, ip = _ip()
        ip._req.see_by = "調停"
        await _returned_see(ip)
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full", effort="low")),
        ) as arb:
            d = await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return d, arb

    d, arb = asyncio.run(scenario())
    assert arb.called and d.effort == "low"


def test_a_see_thrown_by_the_main_llm_skips_the_arbiter() -> None:
    async def scenario():
        a, ip = _ip()
        ip._req.see_by = "主LLM"
        ip._req.see_effort = "medium"
        await _returned_see(ip)
        with patch("familiar_agent.loop.event_loop.arbitrate", new=AsyncMock()) as arb:
            d = await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return d, arb

    d, arb = asyncio.run(scenario())
    assert not arb.called and d.branch == "full" and d.effort == "medium"


def test_the_arbiter_is_told_whether_it_can_see() -> None:
    async def scenario():
        a, ip = _ip()
        a._camera = MagicMock()
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full")),
        ) as arb:
            await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return arb

    arb = asyncio.run(scenario())
    assert arb.call_args.kwargs["can_see"] is True


def test_the_action_branch_records_who_looked() -> None:
    """`(c)` で調停が see を投げると、`see_by` が調停になり lookup が始まる。"""
    import inspect

    src = inspect.getsource(InformationProcessing._iterate)
    assert 'see_by = "調停"' in src or "see_by = '調停'" in src
    src2 = inspect.getsource(InformationProcessing._act_on_decision)
    assert 'see_by = "主LLM"' in src2
