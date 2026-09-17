"""掛けているあいだは聞かない（`TIMER_MIC_CLOSE`・知-o 段 5・2026-09-18・`設計方針_タイマー` v0.6 §9）。

マイクは止めない。書き起こしまでは動かし、**タイマーの操作の言葉だけ通す**（それ以外は捨てて O にも
残さない）。機械が決めるのは「通すか」まで——何をするかは調停でも主LLM でも同じ道具。
聞かない状態は動いているタイマーから導く（新しい状態は持たない）。一時停止中と `/mic on` のあとは聞く。
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import timer_rules
from familiar_agent.realtime_stt_session import RealtimeSttSession

from tests.test_timer_tool import _tool

NOW = datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "true")


# ── 操作の言葉 ────────────────────────────────────────────────────────────


def test_control_words_are_short_utterances_about_the_timer():
    for s in ("止めて", "タイマー止めて", "一時停止", "再開して", "ストップ", "中止", "やめて"):
        assert timer_rules.is_control_word(s), s
    for s in ("こんにちは", "止めてほしいって昨日言ってたよね", "再開発の話だけど", ""):
        assert not timer_rules.is_control_word(s), s


# ── 聞かない状態の導出 ─────────────────────────────────────────────────────


def test_listening_is_closed_only_while_a_timer_runs():
    t, store, _ = _tool()
    assert t.listening_closed() == ""
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert "パスタ" in t.listening_closed()
    asyncio.run(t.call("pause_timer", {"id": 1}))
    assert t.listening_closed() == ""  # 一時停止中は聞く（「やっぱりやめる」が言える）
    asyncio.run(t.call("resume_timer", {"id": 1}))
    assert t.listening_closed() != ""
    asyncio.run(t.call("cancel_timer", {"id": 1}))
    assert t.listening_closed() == ""


def test_a_stopwatch_does_not_close_listening():
    t, _, _ = _tool()
    asyncio.run(t.call("start_stopwatch", {"label": "ランニング"}))
    assert t.listening_closed() == ""


def test_the_flag_off_keeps_listening(monkeypatch):
    monkeypatch.setenv("TIMER_MIC_CLOSE", "false")
    t, _, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert t.listening_closed() == ""


def test_mic_on_reopens_listening_for_that_timer():
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    text, ok = asyncio.run(t.call("listen", {}))
    assert ok and "聞く" in text and t.listening_closed() == ""


# ── 書き起こしの門 ─────────────────────────────────────────────────────────


async def _relay(texts, gate):
    session = RealtimeSttSession("dummy")
    committed_q: asyncio.Queue[str] = asyncio.Queue()
    input_q: asyncio.Queue[str | None] = asyncio.Queue()
    session._incoming_committed = committed_q
    session._committed_queue = input_q
    session.mic_gate = gate
    task = asyncio.create_task(session._committed_relay())
    for s in texts:
        await committed_q.put(s)
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    out = []
    while not input_q.empty():
        out.append(input_q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_only_control_words_pass_while_closed():
    out = await _relay(
        ["こんにちは", "止めて", "今日の天気は", "再開して"], gate=lambda: "タイマー「パスタ」"
    )
    assert out == ["止めて", "再開して"]


@pytest.mark.asyncio
async def test_everything_passes_while_open():
    out = await _relay(["こんにちは", "止めて"], gate=lambda: "")
    assert out == ["こんにちは", "止めて"]


# ── 枠と命令 ──────────────────────────────────────────────────────────────


def test_the_frame_says_not_listening():
    t, _, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert "聞いていない" in t.frame()


def test_the_mic_command_reopens_without_the_llm():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("聞く：id=1 「パスタ」（鳴るまで）", True))
    a._handle_mic_command = lambda ui: Agent._handle_mic_command(a, ui)
    reply = asyncio.run(a._handle_mic_command("/mic on"))
    assert reply and "聞く" in reply
    a._timer_tool.call.assert_awaited_once_with("listen", {})
    assert asyncio.run(a._handle_mic_command("こんにちは")) is None


def test_the_status_card_shows_not_listening():
    from familiar_agent.diagnostics import build_gui_diagnostics

    window = MagicMock()
    window._agent_ready = True
    window._agent_running = False
    window._agent_init_failed = False
    window._agent.mic_gate_reason = lambda: "タイマー「パスタ」"
    window._realtime_stt.gated = False
    snap = build_gui_diagnostics(window)
    assert (
        snap.phase == "not_listening" and "🔇" in snap.headline and "mic=closed" in snap.readiness
    )
