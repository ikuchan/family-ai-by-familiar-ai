"""命令は末尾の区切り文字に寛容（環-o・2026-09-18）。

Mac の Chrome（リモートデスクトップ）→ ibus-mozc → Qt の経路で、Enter の確定時に末尾へ「・」が付く
（`/timer resume・`・3 回とも・09-15 の `/speaker・` と同じ）。GUI のコードに足す経路は無く原因は未特定。
命令として読めず会話入力へ落ちてタイマーの沈黙に飲まれたので、受ける側で末尾の「・」「。」「、」「．」と
空白を無視する（`agent._command_text`・5 命令すべて）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.agent import _command_text


def test_the_tail_is_dropped_but_the_body_is_kept():
    assert _command_text("/timer resume・") == "/timer resume"
    assert _command_text("  /timer stop 3。 ") == "/timer stop 3"
    assert _command_text("/speaker パパ、") == "/speaker パパ"
    assert _command_text("/mic on．・") == "/mic on"
    assert _command_text("/reload・") == "/reload"
    assert _command_text("こんにちは・") == "こんにちは・"  # 命令でなければ触らない


def _agent():
    a = MagicMock(spec=Agent)
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("再開した", True))
    a._alarm_tool = MagicMock()
    a._alarm_tool.call = AsyncMock(return_value=("止めた", True))
    return a


def test_timer_alarm_and_mic_commands_accept_a_trailing_dot():
    a = _agent()
    assert asyncio.run(Agent._handle_timer_command(a, "/timer resume・")) == "再開した"
    a._timer_tool.call.assert_awaited_with("resume_timer", {"id": "all"})
    assert asyncio.run(Agent._handle_timer_command(a, "/timer stop 3・")) == "再開した"
    a._timer_tool.call.assert_awaited_with("cancel_timer", {"id": "3"})
    assert asyncio.run(Agent._handle_alarm_command(a, "/alarm stop・")) == "止めた"
    assert asyncio.run(Agent._handle_mic_command(a, "/mic on・")) == "再開した"
    a._timer_tool.call.assert_awaited_with("listen", {})
    assert asyncio.run(Agent._handle_timer_command(a, "こんにちは・")) is None
