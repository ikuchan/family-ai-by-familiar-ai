"""発話前の検査に、作業状態の過去の記憶の中身を渡す（2026-09-13 実機で露見・記-r で改めた）。

記憶にある天気（11:24 に自分が答えた『明日は晴れ・最高 30℃…』）で答えたら、チェッカーは
「検索も提示も無いのに天気を言った」と `no-invented-knowledge` にした。材料が件数と日付だけで、
中身を渡していなかった。当時は主LLM が使ったと申告した記憶だけを渡していた。記-r（2026-10-14）で
申告は Jev が検査と同じ呼び出しで出すことにしたので、過去の列の中身をまるごと（1 件 300 字まで）渡す。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop.speech_check import facts_ctx, past_rows
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


def test_past_rows_follow_the_map() -> None:
    got = past_rows(_W, {"aaaaaaaaaaaa": "aaaaaaaaaaaa-1"})
    assert got == {"aaaaaaaaaaaa": "自分が答えた：明日9月14日の東京は晴れで、最高30℃、最低22℃。"}


def test_the_facts_carry_the_past_memories() -> None:
    ctx = facts_ctx(saw=False, memories=_W, past=["自分が答えた：明日9月14日の東京は晴れ"])
    assert "作業状態の過去の記憶の中身" in ctx
    assert "明日9月14日の東京は晴れ" in ctx


def test_without_past_memories_the_facts_say_none() -> None:
    ctx = facts_ctx(saw=False, memories=_W, past=[])
    assert "作業状態の過去の記憶の中身：なし" in ctx


def test_the_loop_hands_the_past_memories_to_the_checker() -> None:
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    # 判定は Jev（出-au 段 5-3）。送る文（state）に事実が入っているかを見る。
    a._jev = MagicMock(available=True)
    a._jev.ask = AsyncMock(return_value=None)
    a.config.jev_confidence_min = 0.6
    ip = InformationProcessing(a)

    async def scenario():
        await ip._speech_check_violation("明日は晴れだよ", "", _W, w_id_map=_MAP)
        await ip.close()
        return a._jev.ask.await_args.args[0]

    facts = asyncio.run(scenario())
    assert "最高30℃" in facts
