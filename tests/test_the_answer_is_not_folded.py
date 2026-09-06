"""答えの逐語を畳まない（段 3）。

畳むと想起の母集合から消え、残るのは軽量LLM の一文（80 トークン上限）だけになる。
細部のベクトルが無ければ、細部での近接は起きない（`設計方針_MI間の関係`）。

**片側だけでは確かめたことにならない。** 逐語が残ることと、会話要約も並んで残ることを
対にして見る。片方だけ消えていれば、どちらかが落ちる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent


def _agent():
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
    agent._memory.mark_superseded = MagicMock()
    agent._memory.record_exchange = MagicMock(return_value=1)
    agent._conversation_perspective = MagicMock(return_value={})
    return agent


def _run(agent, **kw):
    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="昨日の天気覚えてる？",
            final_text="晴れてたよ",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
            is_desire_turn=False,
            desires=None,
            **kw,
        )
    )


def test_the_turn_folds_nothing():
    """ターンの後始末で畳む操作を一切しない。"""
    agent = _agent()
    _run(agent, exchange=[("obs1", "起点"), ("obs2", "答え")])
    agent._memory.mark_superseded.assert_not_called()


def test_the_verbatim_and_the_summary_both_survive():
    """逐語は関係の項として残り、会話要約は記録として書かれる。どちらも消えない。"""
    agent = _agent()
    _run(agent, exchange=[("obs1", "起点"), ("obs2", "答え")])

    members = agent._memory.record_exchange.call_args.args[0]
    assert ("obs2", "答え", 1) in members  # 逐語
    assert ("conv-1", "要約", 2) in members  # 要約
    directions = [
        c.kwargs.get("direction") for c in agent._memory.save_async_with_id.call_args_list
    ]
    assert "会話" in directions


def test_the_folding_argument_is_gone():
    """畳む道が残っていると、次に読む人が「まだ畳んでいる」と読む。"""
    import inspect

    sig = inspect.signature(EmbodiedAgent._run_post_response_pipeline)
    assert "superseded_ids" not in sig.parameters
