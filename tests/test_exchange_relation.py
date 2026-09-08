"""一つのターンの記録を、やりとりの関係として順序つきで残す（段 3）。

問いと答えを別々の記憶として置くと、対として想起されない（`設計方針_MI間の関係`）。
起点と版と見た結果と答えの逐語と会話要約を、一つの関係の項として**順序つきで**並べる。

項は時間差で揃う（起点はターン頭、答えは反復の終わり、要約は背景タスク）。全部が揃う
のは `_run_post_response_pipeline` の中なので、関係はそこで一度に書く。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.backends import ToolCall
from tests.test_event_loop import _agent, _run, _turn
from familiar_agent.loop.event_loop import InformationProcessing


def _exchange(a):
    _, kwargs = a._run_post_response_pipeline.call_args
    return list(kwargs.get("exchange") or [])


def test_the_turn_hands_over_its_records_in_order():
    """起点が先、答えが後。順序は関係の `position` になる。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})])])
    _run(a, utterance="今日の天気は？")
    # obs1=起点 / obs2=答えの逐語。
    assert _exchange(a) == [("obs1", "起点"), ("obs2", "答え")]


def test_a_silent_turn_hands_over_no_answer():
    """黙ったターンには答えの項ができない。常に作っていれば、ここが落ちる。"""
    a = _agent(stream_returns=[_turn([], text="")])
    _run(a, utterance="…")
    roles = [r for _, r in _exchange(a)]
    assert "答え" not in roles


def _pipeline_agent():
    agent = MagicMock()
    agent.config = MagicMock()
    agent._emotion_for_turn = AsyncMock(return_value=(None, 0.0, "neutral"))
    agent._update_mood = MagicMock()
    agent._summarize_exchange = AsyncMock(return_value="summary")
    agent._maybe_update_self_narrative = AsyncMock()
    agent._maybe_adapt_values = AsyncMock()
    agent._maybe_discharge_satisfied_drives = AsyncMock()
    agent._active_memory = MagicMock(return_value=agent._memory)
    agent._memory.save_async_with_id = AsyncMock(return_value=("conv-1", True))
    agent._memory.record_exchange = MagicMock(return_value=1)
    agent._conversation_perspective = MagicMock(return_value={})
    return agent


def test_the_summary_is_appended_as_the_last_member():
    """会話要約は最後に来る。背景で遅れて作られるが、位置は末尾で決まっている。"""
    agent = _pipeline_agent()

    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="今日の天気は？",
            final_text="晴れだよ",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
            is_desire_turn=False,
            desires=None,
            exchange=[("obs1", "起点"), ("obs2", "答え")],
        )
    )

    members = agent._memory.record_exchange.call_args.args[0]
    assert members == [
        ("obs1", "起点", 0),
        ("obs2", "答え", 1),
        ("conv-1", "要約", 2),
    ]


def test_no_relation_is_written_when_the_turn_left_nothing():
    """項が要約だけなら、やりとりとは呼べない。関係を書かない。"""
    agent = _pipeline_agent()

    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="",
            final_text="ひとりごと",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
            is_desire_turn=False,
            desires=None,
            exchange=None,
        )
    )

    agent._memory.record_exchange.assert_not_called()


def test_an_interrupted_turn_does_not_leak_into_the_next_one():
    """話しかけられて調べかけを打ち切ったら、そこで一つのやりとりが閉じる。

    閉じないと、打ち切られた問いと新しい問いが**一つのやりとり**に入る（起点が2つ）。
    母集合への持ち越しは別で、打ち切りの記録は次のターンの WR にも載り続ける。
    """
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "明日は晴れだよ"})]),
        ]
    )

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("昨日の天気覚えてる？")
        # 調べかけの途中で話しかける。
        await ip.run_iteration("それより明日の予定は？")

    asyncio.run(scenario())

    # 打ち切りで閉じたやりとりに、新しい問いは入っていない。
    aborted = a._memory.record_exchange.call_args.args[0]
    assert [r for _, r, _ in aborted].count("起点") == 1, aborted

    # 続くターンのやりとりにも、起点は1つだけ。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert [r for _, r in kwargs["exchange"]].count("起点") == 1, kwargs["exchange"]
    # 母集合へは、打ち切りの分も持ち越して渡る。
    assert "obs1" in kwargs["extra_cooccurring_ids"]
