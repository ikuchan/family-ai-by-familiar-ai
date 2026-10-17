"""確認への「はい」「いいえ」は、調停がそのまま道具を投げる（知-o の残り・2026-10-17）。

10/17 17:58 に、確認の問いへの「はい」を Jev が「掛ける」と読んだのに、決定に見出し（`query`）が無く、
`_close_early` を通らずに主LLM へ回った。主LLM は道具を呼ばず「手続きを進めています」と言い、タイマーは掛からなかった。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.arbiter import CONFIRM_QUERY, Decision
from familiar_agent.loop.iteration import Iteration


def _iteration(decision: Decision) -> tuple[Iteration, MagicMock]:
    ip = MagicMock()
    ip._request_generation = 0
    ip._req.trigger_kind = "発話"
    ip._req.cue = "はい"
    ip._dispatch_arbiter_action = AsyncMock()
    it = Iteration.__new__(Iteration)
    it.ip, it.gen, it.utterance = ip, 0, "はい"
    it.decision, it.capped, it.chain, it.max_chain = decision, False, 1, 5
    it.ws = MagicMock(memories=[], returned_actions=frozenset())
    return it, ip


@pytest.mark.parametrize("action", ["confirm", "decline"])
def test_the_answer_is_thrown_by_the_arbiter(action):
    it, ip = _iteration(Decision(branch="action", action=action, query=CONFIRM_QUERY[action]))
    assert asyncio.run(it._close_early()) == ""  # 閉じた（主LLM へは回らない）
    ip._dispatch_arbiter_action.assert_awaited_once()


def test_without_the_heading_it_falls_to_the_main_llm():
    """**反証の側**：見出しが空の決定は主LLM へ回る（10/17 17:58 の形）。試験がその差を見分けている。"""
    it, ip = _iteration(Decision(branch="action", action="confirm"))
    assert asyncio.run(it._close_early()) is None
    ip._dispatch_arbiter_action.assert_not_awaited()
