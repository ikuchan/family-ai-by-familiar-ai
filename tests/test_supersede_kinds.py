"""畳む書き手が、それぞれの理由を種類として渡すことを確かめる（段 2）。

一つの列に四つの意味が乗っていたのを関係へ移した（`設計方針_MI間の関係` v0.3）。
**隠すかどうかは役割 `旧` が決め、種類は理由を言うだけである。** 種類が全部同じなら
移した意味が無いので、書き手ごとに違う種類が渡ることをここで押さえる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.store.relations import KIND_ADVANCE, KIND_REVISION


def _agent() -> MagicMock:
    a = MagicMock()
    a._memory = MagicMock()
    a._memory.mark_superseded = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("new-1", True))
    a._observation_perspective = MagicMock(return_value={})
    a.config.completion_content_max = 500
    return a


def test_advancing_the_chain_says_it_is_a_step_forward() -> None:
    a = _agent()
    ip = InformationProcessing(a)
    ip._chain_head_id = "old-1"
    ip._advance_chain("new-1", "内容")
    assert a._memory.mark_superseded.call_args.kwargs["kind"] == KIND_ADVANCE


def test_a_new_version_says_it_is_a_revision() -> None:
    a = _agent()
    ip = InformationProcessing(a)
    ip._version_id = "ver-1"

    asyncio.run(ip._write_version())

    kinds = [c.kwargs.get("kind") for c in a._memory.mark_superseded.call_args_list]
    assert KIND_REVISION in kinds
