"""直近のやりとりを、ターンの文脈へ渡す（段 4）。

いま主LLM へ渡るのは現在の発話一通だけで、会話履歴はどこにもない。継起をさかのぼって
組み立てたものを可変部へ載せる。

**逐語で出す。** 細部が要るからこの設計にしたので、ここで縮めると意味がない。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from unittest.mock import AsyncMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _run, _turn

_NOW = datetime(2026, 9, 7, 21, 14, tzinfo=timezone.utc)


def _rows():
    return [
        {
            "content": "明日の運動会って何時から？",
            "role": "起点",
            "direction": "発話",
            "timestamp": _NOW,
            "depth": 1,
        },
        {
            "content": "8時半に開会式だよ。",
            "role": "答え",
            "direction": "発話",
            "timestamp": _NOW,
            "depth": 1,
        },
    ]


def test_the_recent_talk_reaches_the_system_text():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(return_value="m1")
    a._memory.recent_exchanges = MagicMock(return_value=_rows())
    _run(a, utterance="開会式って何時だっけ")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "直近のやりとり" in system
    assert "明日の運動会って何時から？" in system
    assert "8時半に開会式だよ。" in system


def test_the_verbatim_is_not_shortened():
    """W は 120 字で切るが、ここは切らない。切ると細部が消える。"""
    long = "あ" * 400
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(return_value="m1")
    a._memory.recent_exchanges = MagicMock(
        return_value=[
            {"content": long, "role": "答え", "direction": "発話", "timestamp": _NOW, "depth": 0},
        ]
    )
    _run(a, utterance="ねえ")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert long in system


def test_nothing_is_shown_when_there_is_no_chain():
    """空の見出しは「無い」ではなく「調べたが無い」と読まれる。出さない。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(return_value="m1")
    a._memory.recent_exchanges = MagicMock(return_value=[])
    _run(a, utterance="はじめまして")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "直近のやりとり" not in system


def test_the_walk_starts_from_the_last_closed_exchange():
    """このターンの起点からは辿れない。まだどのやりとりにも属していないからである。

    起動直後は持ち回りが空なので、DB から一度だけ引く。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._memory.latest_exchange_origin = MagicMock(return_value="前回の起点")
    a._evaluator.judge_follows = AsyncMock(return_value="m1")
    a._memory.recent_exchanges = MagicMock(return_value=[])
    _run(a, utterance="ねえ")
    assert a._memory.recent_exchanges.call_args.args[0] == "前回の起点"
    assert a._memory.latest_exchange_origin.call_count == 1


def test_the_cursor_moves_to_the_exchange_that_just_closed():
    """一つ閉じたら、次のターンはそこから見せる。DB へは二度引きにいかない。"""
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "うん"})]),
            _turn([ToolCall(id="s2", name="say", input={"text": "はい"})]),
        ]
    )
    a._memory.latest_exchange_origin = MagicMock(return_value=None)
    a._evaluator.judge_follows = AsyncMock(return_value="m1")
    a._memory.recent_exchanges = MagicMock(return_value=[])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("ひとつめ")
        await ip.run_iteration("ふたつめ")

    asyncio.run(scenario())
    # ふたつめのターンは、ひとつめの起点（obs1）から見せる。
    assert a._memory.recent_exchanges.call_args.args[0] == "obs1"
    assert a._memory.latest_exchange_origin.call_count == 1


def test_nothing_is_shown_when_this_turn_continues_nothing():
    """判定が続き先を返さなければ、直近のやりとりを載せない（`根拠台帳` §29）。

    載せると、関係のない会話が文脈に混ざる。新しい話の始まりに前の話は要らない。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(return_value=None)
    a._memory.latest_exchange_origin = MagicMock(return_value="前回の起点")
    a._memory.recent_exchanges = MagicMock(return_value=_rows())
    _run(a, utterance="はじめまして")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "直近のやりとり" not in system
    a._memory.recent_exchanges.assert_not_called()
