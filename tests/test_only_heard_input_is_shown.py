"""会話ログに出すのは受けた入力だけ（出-au 段 1-4・2026-09-26・`設計方針_判定の段` §2 原則 2）。

以前は画面が、書き起こしが確定した時点・打った時点で会話ログに `[話者] 本文` を出していた。窓の外で捨てる声
（テレビ・家族の話）も、画面では「聞いた」ように見えた。

- ループの門が「受けた／捨てた」を知らせ（`set_heard_listener`）、画面はそれだけで出し分ける。
- 捨てた入力は、状態の行に「🎤 聞いていない：…」を**次の入力か状態が来るまで**出す（本人の決定 ウ）。
- コマンド（`/speaker` など）はループへ届く前に `agent.run` が返すので、そこから「受けた」を知らせる。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

pytestmark = pytest.mark.real_window  # 門そのものの知らせを見る


def _ip():
    a = _agent(stream_returns=[])
    a._nudge_seeking = AsyncMock()
    a._social_presence_permission = MagicMock(return_value=1.0)
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None
    ip._ensure_driver = lambda: None
    ip._abort_lookups = AsyncMock()
    heard: list = []
    ip.set_heard_listener(lambda text, ok: heard.append((str(text), ok)))
    return ip, heard


def _push(ip, text):
    async def go():
        task = asyncio.ensure_future(ip.push_utterance(text, source="voice"))
        await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_the_gate_tells_what_it_dropped():
    ip, heard = _ip()
    _push(ip, "ごちそうさまでした")
    assert heard == [("ごちそうさまでした", False)]


def test_the_gate_tells_what_it_heard():
    ip, heard = _ip()
    _push(ip, "パジュ、おはよう")
    ip._wake_window().open(time.monotonic())
    _push(ip, "うん")
    assert heard == [("パジュ、おはよう", True), ("うん", True)]


def test_a_command_is_shown_as_heard():
    """`/speaker` などはループへ届かないので、`agent.run` が「受けた」を知らせる。"""
    from familiar_agent.agent import EmbodiedAgent as Agent
    from tests.test_input_commands_before_loop_branch import _agent as _cmd_agent

    a = _cmd_agent()
    heard: list = []
    a._heard_listener = lambda text, ok: heard.append((str(text), ok))
    asyncio.run(Agent.run(a, "/speaker パパ"))
    assert heard == [("/speaker パパ", True)]
    a._info_processing.push_utterance.assert_not_awaited()


def test_the_agent_hands_the_listener_to_the_loop():
    from familiar_agent.agent import EmbodiedAgent as Agent

    src = inspect.getsource(Agent.set_heard_listener)
    assert "set_heard_listener" in src and "_heard_listener" in src


@pytest.mark.parametrize(
    "module, gone",
    [
        ("familiar_agent.gui", 'self._log.append_line(f"[{self._get_active_speaker()}] {spoken}")'),
        (
            "familiar_agent.gui",
            'self._log.append_line(f"[{self._get_active_speaker()}] {text}")\n        self._input_queue',
        ),
    ],
)
def test_the_gui_no_longer_logs_input_before_the_gate(module, gone):
    pytest.importorskip("PySide6")
    import importlib

    src = inspect.getsource(importlib.import_module(module))
    assert gone not in src
    assert "聞いていない：" in src and "set_heard_listener(self._on_heard)" in src


def test_the_tui_no_longer_logs_input_before_the_gate():
    import importlib

    src = inspect.getsource(importlib.import_module("familiar_agent.tui"))
    assert "聞いていない：" in src and "set_heard_listener(self._on_heard)" in src
    assert "self._log_user(text)\n        self._last_interaction" not in src
