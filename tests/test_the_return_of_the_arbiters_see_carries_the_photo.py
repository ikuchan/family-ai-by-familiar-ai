"""調停が投げた see の帰りでは、ループが写真を調停へ渡す（`イベント駆動ループ` v0.46）。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop.arbiter import Decision as ArbiterDecision
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent


def _recalled(obs_id, image_path):
    mi = MI(
        id=obs_id,
        content="見た",
        timestamp=datetime.now(),
        direction="観察",
        obs_id=obs_id,
        image_path=image_path,
    )
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


async def _returned_see(ip):
    ip._req.lookups.append(Lookup(index=1, action="see", query="目の前を見る", generation=0))
    ip._triggers.put_nowait(
        Trigger(kind="完了", query="目の前を見る", result="（見えた）", index=1)
    )
    await ip._intake()


def test_the_photo_goes_to_the_arbiter_when_it_looked(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")

    async def scenario():
        a = _agent(stream_returns=[])
        ip = InformationProcessing(a)
        ip._req.see_by = "調停"
        ip._req.turn_records = [("起点", "起点"), ("見た1", "見た")]
        await _returned_see(ip)
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="light", text="机")),
        ) as arb:
            await ip._decide(
                utterance="x",
                workspace_ctx="",
                present_ctx="",
                capped=False,
                round_=1,
                memories=[_recalled("見た1", str(path))],
            )
        await ip.close()
        return arb

    arb = asyncio.run(scenario())
    assert arb.call_args.kwargs["image_b64"] == "SlBFRw=="


def test_no_photo_is_sent_when_nothing_was_seen() -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        ip = InformationProcessing(a)
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full")),
        ) as arb:
            await ip._decide(
                utterance="x", workspace_ctx="", present_ctx="", capped=False, round_=1, memories=[]
            )
        await ip.close()
        return arb

    arb = asyncio.run(scenario())
    assert not arb.call_args.kwargs.get("image_b64")


def test_the_decision_is_logged_at_info(caplog) -> None:
    """何を選んだかが INFO で残る（出-k-い の材料。これまでは DEBUG で見えなかった）。"""
    import logging

    async def scenario():
        a = _agent(stream_returns=[])
        ip = InformationProcessing(a)
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="light", text="机", effort="low")),
        ):
            with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
                await ip._decide(
                    utterance="x",
                    workspace_ctx="",
                    present_ctx="",
                    capped=False,
                    round_=1,
                    memories=[],
                )
        await ip.close()

    asyncio.run(scenario())
    assert any("調停=light" in r.getMessage() for r in caplog.records)
