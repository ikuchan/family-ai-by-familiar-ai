"""つなぎは疑問文にしない（2026-09-13 実機で露見）。

調停が「調べて」に対して検索を投げながら、つなぎに「わかりました、何を調べましょうか？」と
書いた。聞き返しながら調べる矛盾で、相手は答えるべきか待つべきか分からない。規則
（内容に触れず、これから調べると伝えるだけ）は文で頼んでいたが、機械の守りが無かった。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._dif = MagicMock(speak=AsyncMock())
    return ip


def test_a_question_mark_filler_is_dropped() -> None:
    ip = _ip()

    async def scenario():
        await ip._say_filler("わかりました、何を調べましょうか？")
        await ip._say_filler("What should I look up?")
        n = ip._dif.speak.await_count
        await ip.close()
        return n

    assert asyncio.run(scenario()) == 0


def test_a_plain_filler_still_goes_out() -> None:
    ip = _ip()

    async def scenario():
        await ip._say_filler("調べてみますね。")
        # 声は背景で鳴る（出-aq 段 1）。数える前に 1 拍おいて、背景の声を走らせる。
        await asyncio.sleep(0)
        n = ip._dif.speak.await_count
        await ip.close()
        return n

    assert asyncio.run(scenario()) == 1
