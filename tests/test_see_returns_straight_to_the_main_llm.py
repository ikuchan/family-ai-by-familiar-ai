"""see の帰りは調停を飛ばして主LLM へ戻す（`イベント駆動ループ` v0.43）。

見ると決めたのは主LLM 自身で、画像を受け取るのも主LLM だけ。軽量LLM に判定し直させると
`light` や `recall` へ逸れて、画像を見ずに答える（2026-09-12 実機）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    return a, InformationProcessing(a)


def test_after_a_see_completion_the_arbiter_is_skipped() -> None:
    async def scenario():
        a, ip = _ip()
        ip._req.see_effort = "medium"
        ip._req.lookups.append(Lookup(index=1, action="see", query="目の前を見る", generation=0))
        ip._triggers.put_nowait(
            Trigger(kind="完了", query="目の前を見る", result="（見えた）", index=1)
        )
        await ip._intake()
        with patch("familiar_agent.loop.event_loop.arbitrate", new=AsyncMock()) as arb:
            d = await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return d, arb

    d, arb = asyncio.run(scenario())
    assert not arb.called
    assert d.branch == "full" and d.effort == "medium"


def test_the_shortcut_is_used_once() -> None:
    async def scenario():
        a, ip = _ip()
        ip._req.lookups.append(Lookup(index=1, action="see", query="目の前を見る", generation=0))
        ip._triggers.put_nowait(
            Trigger(kind="完了", query="目の前を見る", result="（見えた）", index=1)
        )
        await ip._intake()
        with patch("familiar_agent.loop.event_loop.arbitrate", new=AsyncMock()) as arb:
            await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
            await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=2
            )
        await ip.close()
        return arb

    arb = asyncio.run(scenario())
    assert arb.call_count == 1


def test_a_recall_completion_still_goes_through_the_arbiter() -> None:
    async def scenario():
        a, ip = _ip()
        ip._req.lookups.append(Lookup(index=1, action="recall", query="昨日", generation=0))
        ip._triggers.put_nowait(Trigger(kind="完了", query="昨日", result="…", index=1))
        await ip._intake()
        with patch("familiar_agent.loop.event_loop.arbitrate", new=AsyncMock()) as arb:
            await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return arb

    assert asyncio.run(scenario()).called


def test_the_main_llm_effort_is_remembered_when_it_asks_to_see() -> None:
    import inspect

    src = inspect.getsource(InformationProcessing._act_on_decision)
    assert 'lookup_tc.name == "see"' in src and "see_effort" in src
