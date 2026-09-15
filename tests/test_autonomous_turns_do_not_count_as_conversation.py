"""自発ターンは「人と話した」に数えない（情-g・2026-09-15 実機）。

`seeking` が 5 分おきに 18 回・毎回「ひとり 1 回目」。pipeline が毎ターン `_last_human_at` と
`record_conversation()` を書き、ループは情動が起点でも cue を `user_input` として渡していたため、
ひとりの回数（倍々に伸ばす仕組み・情-d）が自発ターンのたびに 0 へ戻っていた。
人の発話の印は入口（`push_utterance`）が付ける。pipeline は起点が人の発話のときだけ関係を記録する。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _WAIT_TICKS, ToolCall, _agent, _turn


def _pipeline_kwargs(a) -> dict:
    for _ in range(_WAIT_TICKS):
        if a._run_post_response_pipeline.call_args:
            return a._run_post_response_pipeline.call_args.kwargs
    return a._run_post_response_pipeline.call_args.kwargs


def test_an_affect_turn_tells_the_pipeline_it_was_not_a_human():
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
    assert a._run_post_response_pipeline.call_args.kwargs["human"] is False
    assert a._last_human_at == 1000.0  # 自発ターンは人の印を動かさない


def test_a_human_turn_tells_the_pipeline_it_was_a_human():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.push_utterance("こんにちは", on_text=lambda _t: None)
        for _ in range(_WAIT_TICKS):
            if a._run_post_response_pipeline.call_args:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert a._run_post_response_pipeline.call_args.kwargs["human"] is True


def _bare_agent():
    agent = MagicMock()
    agent._relationship = MagicMock()
    agent._summarize_exchange = AsyncMock(return_value="")
    agent._emotion_for_turn = AsyncMock(return_value=(None, 0.0, "neutral"))
    agent._maybe_discharge_satisfied_drives = AsyncMock()
    agent._record_cooccurrence = MagicMock()
    agent._oif.extend = MagicMock()
    agent._oif.recall = AsyncMock(return_value=[])
    agent._oif.write = AsyncMock(return_value="conv-1")
    agent._last_human_at = 1000.0
    return agent


def _run(agent, *, human: bool):
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
            human=human,
        )
    )


def test_the_pipeline_records_a_conversation_only_for_a_human_origin():
    agent = _bare_agent()
    _run(agent, human=False)
    agent._relationship.record_conversation.assert_not_called()
    assert agent._last_human_at == 1000.0
    agent = _bare_agent()
    _run(agent, human=True)
    agent._relationship.record_conversation.assert_called_once()
    assert agent._last_human_at == 1000.0  # 印は入口が付ける。pipeline は書かない
