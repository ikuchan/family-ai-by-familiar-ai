"""自発ターンは「人と話した」に数えない（情-g・2026-09-15 実機）。

`seeking` が 5 分おきに 18 回・毎回「ひとり 1 回目」。pipeline が毎ターン `_last_human_at` を
書き、ループは情動が起点でも cue を `user_input` として渡していたため、ひとりの回数（倍々に
伸ばす仕組み・情-d）が自発ターンのたびに 0 へ戻っていた。人の発話の印は入口（`push_utterance`）
だけが付け、pipeline は書かない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _WAIT_TICKS, ToolCall, _agent, _turn


def test_an_affect_turn_does_not_move_the_human_mark():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "ひとりごと"})])])
    a._last_human_at = 1000.0

    async def scenario():
        ip = InformationProcessing(a)
        ip.set_output(lambda _t: None)
        ip.start()
        ip.push_affect("SEEKING", "何かを知りたい")
        for _ in range(_WAIT_TICKS):
            if a._run_post_response_pipeline.call_args:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert a._run_post_response_pipeline.call_args is not None
    assert a._last_human_at == 1000.0  # 自発ターンは人の印を動かさない


def _bare_agent():
    agent = MagicMock()
    agent._summarize_exchange = AsyncMock(return_value="")
    agent._emotion_for_turn = AsyncMock(return_value=(None, 0.0, "neutral"))
    agent._maybe_discharge_satisfied_drives = AsyncMock()
    agent._record_cooccurrence = MagicMock()
    agent._oif.extend = MagicMock()
    agent._oif.recall = AsyncMock(return_value=[])
    agent._oif.write = AsyncMock(return_value="conv-1")
    agent._last_human_at = 1000.0
    return agent


def _run(agent):
    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="[内的な促し:SEEKING] 探索したい",
            final_text="ひとりごと",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
        )
    )


def test_the_pipeline_never_writes_the_human_mark():
    agent = _bare_agent()
    _run(agent)
    assert agent._last_human_at == 1000.0  # 印は入口が付ける。pipeline は書かない
