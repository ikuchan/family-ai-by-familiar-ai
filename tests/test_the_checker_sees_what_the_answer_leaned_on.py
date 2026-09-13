"""整合チェックに、主LLM が使ったと申告した記憶の中身を渡す（2026-09-13 実機で露見）。

記憶にある天気（11:24 に自分が答えた『明日は晴れ・最高 30℃…』）で答えたら、チェッカーは
「検索も提示も無いのに天気を言った」と `no-invented-knowledge` にした。材料が件数と日付だけで、
中身を渡していなかった。主LLM は `memory_verdicts` で使った記憶を申告しているので、
`referred`／`important` の行をそのまま渡す。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop.coherence import facts_ctx, used_lines
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _recalled(obs_id, content):
    mi = MI(id=obs_id, content=content, timestamp=datetime.now(), direction="発話", obs_id=obs_id)
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


_W = [
    _recalled("aaaaaaaaaaaa-1", "自分が答えた：明日9月14日の東京は晴れで、最高30℃、最低22℃。"),
    _recalled("bbbbbbbbbbbb-1", "相手が言った：おはよう"),
]
_MAP = {"aaaaaaaaaaaa": "aaaaaaaaaaaa-1", "bbbbbbbbbbbb": "bbbbbbbbbbbb-1"}


def test_used_lines_picks_referred_and_important_by_the_map() -> None:
    raw = [
        {"id": "aaaaaaaaaaaa", "verdict": "referred"},
        {"id": "bbbbbbbbbbbb", "verdict": "unused"},
    ]
    got = used_lines(raw, _MAP, _W)
    assert got == ["自分が答えた：明日9月14日の東京は晴れで、最高30℃、最低22℃。"]


def test_the_facts_carry_the_used_memories() -> None:
    ctx = facts_ctx(saw=False, memories=_W, used=["自分が答えた：明日9月14日の東京は晴れ"])
    assert "主LLM が使ったと申告した記憶" in ctx
    assert "明日9月14日の東京は晴れ" in ctx


def test_without_verdicts_the_facts_say_none_declared() -> None:
    ctx = facts_ctx(saw=False, memories=_W, used=[])
    assert "申告なし" in ctx


def test_the_loop_hands_the_used_lines_to_the_checker() -> None:
    a = _agent(stream_returns=[])
    a.config.coherence_check = True
    a._evaluator.check_response_coherence = AsyncMock(return_value=None)
    ip = InformationProcessing(a)

    async def scenario():
        await ip._coherence_violation(
            "明日は晴れだよ",
            "",
            _W,
            verdicts=[{"id": "aaaaaaaaaaaa", "verdict": "important"}],
            w_id_map=_MAP,
        )
        await ip.close()
        return a._evaluator.check_response_coherence.call_args.kwargs["facts"]

    facts = asyncio.run(scenario())
    assert "最高30℃" in facts
