"""調停は直近のやりとりを見て判断する（2026-09-13 実機で露見）。

「明日の天気は？」→（検索が空振り）→「調べて」で、調停は直近のやりとりを渡されておらず、
「調べて」を新しい検索（今日のニュース）にした。主LLM だけが（続きと判定されたとき）直近を
受け取っていた。調停には**最新 2 往復を無条件で**渡す。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.io.oif import Said
from familiar_agent.loop.arbiter import Decision as ArbiterDecision, arbitrate
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_the_prompt_carries_the_recent_exchange_when_given() -> None:
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="調べて",
            workspace_ctx="",
            recent_ctx="[直近のやりとり（古い順）]\n- 09:46 相手：明日の天気は？\n- 09:46 わたし：気象庁のサイトは見つかったんですが",
        )
    )
    prompt = b.complete.call_args.args[0]
    assert "明日の天気は？" in prompt
    assert prompt.index("直近のやりとり") < prompt.index("[人の言葉]")


def test_the_loop_hands_the_last_two_exchanges_to_the_arbiter() -> None:
    a = _agent(stream_returns=[])
    a._oif.latest_origin = MagicMock(return_value="起点9")  # type: ignore[method-assign]
    a._oif.exchanges = MagicMock(  # type: ignore[method-assign]
        return_value=[
            Said(content="おはよう", role="起点", when=datetime.now(), depth=2),
            Said(content="おはよう！", role="答え", when=datetime.now(), depth=2),
            Said(content="今日は暑いね", role="起点", when=datetime.now(), depth=1),
            Said(content="ほんとだね", role="答え", when=datetime.now(), depth=1),
            Said(content="明日の天気は？", role="起点", when=datetime.now(), depth=0),
            Said(content="調べてみますね", role="つなぎ", when=datetime.now(), depth=0),
            Said(
                content="気象庁のサイトは見つかったんですが",
                role="答え",
                when=datetime.now(),
                depth=0,
            ),
        ]
    )
    ip = InformationProcessing(a)

    async def scenario():
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full")),
        ) as arb:
            await ip._decide(
                utterance="調べて",
                workspace_ctx="",
                present_ctx="",
                capped=False,
                round_=1,
                memories=[],
            )
        await ip.close()
        return arb.call_args.kwargs.get("recent_ctx", "")

    recent = asyncio.run(scenario())
    assert "明日の天気は？" in recent and "気象庁" in recent and "今日は暑いね" in recent
    assert "おはよう" not in recent, "最新 2 往復だけ"
