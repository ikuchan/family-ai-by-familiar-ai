"""主LLM の決定を実行する部分を切り出す（環-h・段ろ）。

`_iterate` は、主LLM を呼んだあと **その返りをどう実行するか**を 60 行かけて捌いていた。
出口は4つある（世代ずれ／道具投げ／発話／素テキスト）。

環-h では、主LLM は投げっぱなしになり、**返りは別の反復（出す反復）で実行される**。実行の
部分をいま切り出しておけば、**h-は で呼び元が変わるだけ**になり、挙動が変わったときの原因が
そこだけに絞れる。

**この段では挙動を変えない。** 同期のまま、`_iterate` から呼ぶ。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.backends.types import TurnResult
from familiar_agent.loop.event_loop import InformationProcessing


def _ip(gen: int = 0):
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._request_generation = gen
    ip._cue = "手がかり"
    ip._say_filler = AsyncMock()
    ip._start_lookup = MagicMock()
    ip._apply_memory_verdicts = MagicMock()
    ip._coherence_violation = AsyncMock(return_value=None)
    ip._speak = AsyncMock(return_value=("はい", "発話"))
    ip._finish = AsyncMock()
    ip._emit = MagicMock()
    ip._tools = MagicMock(return_value=[])
    a = MagicMock()
    a.backend.stream_turn = AsyncMock()
    a.config.max_tokens = 1000
    ip._agent = a
    return ip, a


def _turn(*calls: ToolCall, text: str = "") -> TurnResult:
    return TurnResult(stop_reason="tool_use", text=text, tool_calls=list(calls))


def _run(ip, result, *, gen=0, memories=None, capped=False):
    return asyncio.run(
        ip._act_on_decision(
            result,
            memories=memories if memories is not None else [],
            recent_ctx="",
            utterance="こんばんは",
            system=("安定", "可変"),
            effort="high",
            gen=gen,
            capped=capped,
        )
    )


# ── 4つの出口 ──────────────────────────────────────────────────────────────


def test_a_stale_request_is_dropped():
    """打ち切られた求めの返りは、出力せずに畳む。"""
    ip, _a = _ip(gen=3)
    assert _run(ip, _turn(ToolCall("t", "say", {"text": "はい"})), gen=1) == ""
    ip._speak.assert_not_awaited()
    ip._finish.assert_not_awaited()


def test_a_tool_call_is_dispatched_and_the_turn_stays_open():
    ip, _a = _ip()
    got = _run(ip, _turn(ToolCall("t", "search_deferred", {"query": "明日の天気"})))
    assert got == ""
    ip._start_lookup.assert_called_once()
    ip._finish.assert_not_awaited()  # 閉じない


def test_a_say_speaks_and_closes():
    ip, _a = _ip()
    got = _run(ip, _turn(ToolCall("t", "say", {"text": "はい"})), memories=[{"memory_id": "m1"}])
    assert got == "はい"
    ip._speak.assert_awaited_once_with("はい")
    ip._finish.assert_awaited_once_with("はい", [{"memory_id": "m1"}], "発話")


def test_plain_text_is_shown_but_not_spoken():
    """`say` を呼ばなければ声にならない。閉じるが結末は沈黙（`独白` として残る）。"""
    ip, _a = _ip()
    got = _run(ip, _turn(text="なるほどと思った"))
    assert got == "なるほどと思った"
    ip._emit.assert_called_once_with("なるほどと思った")
    ip._speak.assert_not_awaited()
    ip._finish.assert_awaited_once_with("なるほどと思った", [], "沈黙")


# ── 併記と上限 ─────────────────────────────────────────────────────────────


def test_a_say_alongside_a_tool_becomes_a_filler():
    """発話と動作が一緒に来たら、発話はつなぎとして出し、出力は動作とする。"""
    ip, _a = _ip()
    _run(
        ip,
        _turn(
            ToolCall("s", "say", {"text": "調べるね"}), ToolCall("t", "recall", {"query": "昨日"})
        ),
    )
    ip._say_filler.assert_awaited_once_with("調べるね")
    ip._start_lookup.assert_called_once()


def test_at_the_cap_a_tool_call_is_ignored():
    """上限の反復では調べる動作を渡していないので、返ってきても投げない。"""
    ip, _a = _ip()
    _run(ip, _turn(ToolCall("t", "recall", {"query": "昨日"})), capped=True)
    ip._start_lookup.assert_not_called()


# ── 整合チェックの差し戻し ──────────────────────────────────────────────────


def test_a_violation_sends_it_back_once():
    ip, a = _ip()
    ip._coherence_violation = AsyncMock(return_value="見ていないのに見たと言っている")
    a.backend.stream_turn = AsyncMock(
        return_value=(_turn(ToolCall("r", "say", {"text": "直した"})), None)
    )
    _run(ip, _turn(ToolCall("t", "say", {"text": "そこに本があるね"})))
    a.backend.stream_turn.assert_awaited_once()
    ip._speak.assert_awaited_once_with("直した")


# ── 切り出せていること ──────────────────────────────────────────────────────


def test_the_iteration_delegates_instead_of_doing_it_itself():
    src = inspect.getsource(InformationProcessing._iterate)
    assert "self._act_on_decision(" in src
    assert "[SELF-CHECK]" not in src  # 差し戻しは切り出した側にある
