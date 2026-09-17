"""一時停止・再開（知-o 段 4・2026-09-18・`設計方針_タイマー` v0.5 §12）。

何度止めても再開しても合うように、`paused_at`（いま止めている時刻）と `paused_total_sec`（止めていた
累計）の 2 列で持つ（066）。残り＝`due + 累計 − now`、止めている間は `due + 累計 − paused_at`。
鳴らす側は一時停止中を拾わない。道具は調停の候補と主LLM の道具の**両方**、`/timer pause`／`/timer resume` も。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.core import timer_rules
from familiar_agent.core.silence_hold import lifts
from familiar_agent.loop import timer_watch
from familiar_agent.store.timers import TimerStore

from tests.test_timer_tool import _tool

NOW = datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc)
M = timedelta(minutes=1)


@pytest.fixture(autouse=True)
def _confirm_off(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "false")


# ── 純関数 ─────────────────────────────────────────────────────────────────


def _row(**kw):
    base = {
        "id": 1,
        "label": "パスタ",
        "due": NOW + 3 * M,
        "started_at": NOW,
        "paused_at": None,
        "paused_total_sec": 0.0,
    }
    base.update(kw)
    return base


def test_remaining_counts_down_while_running():
    assert timer_rules.remaining(_row(), NOW + 1 * M) == 2 * M


def test_remaining_stands_still_while_paused_and_survives_three_cycles():
    # 1 分走って止め（残り 2:00）、30 秒止めて再開、また 1 分走って止め、…
    r = _row(paused_at=NOW + 1 * M)
    assert timer_rules.remaining(r, NOW + 5 * M) == 2 * M  # 止めている間は動かない
    r = _row(paused_at=None, paused_total_sec=30.0, due=NOW + 3 * M + timedelta(seconds=30))
    assert timer_rules.remaining(r, NOW + 2 * M) == timedelta(seconds=90)


def test_the_frame_says_paused():
    text = timer_rules.render_frame([_row(paused_at=NOW + 1 * M)], [], now=NOW + 2 * M)
    assert "一時停止中" in text and "残り 2:00" in text


def test_pause_and_resume_words_pass_the_silence_gate():
    assert lifts("会話入力", "一時停止", speaker="パパ", asker="パパ")
    assert lifts("会話入力", "再開して", speaker="パパ", asker="パパ")
    assert not lifts("会話入力", "再開して", speaker="たいき", asker="パパ")


# ── 器（実 DB）────────────────────────────────────────────────────────────


def _store() -> TimerStore:
    conn = psycopg2.connect(
        os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor
    )
    conn.autocommit = True
    return TimerStore(conn)


def test_the_store_pauses_resumes_and_extends_due_each_time():
    s = _store()
    tid = s.add(
        label="パスタ", due=NOW + 3 * M, asked_by="パパ", obs_id=None, passes_quiet=False, now=NOW
    )
    assert s.pause(tid, now=NOW + 1 * M) is True
    assert s.pause(tid, now=NOW + 1 * M) is False  # 二重には止めない
    assert s.due_now(now=NOW + 10 * M) == []  # 止めている間は鳴らない
    assert s.resume(tid, now=NOW + 2 * M) is True  # 1 分止めていた
    r = next(x for x in s.active(now=NOW + 2 * M) if x["id"] == tid)
    assert r["paused_at"] is None and r["paused_total_sec"] == 60.0 and r["due"] == NOW + 4 * M
    assert s.pause(tid, now=NOW + 3 * M) and s.resume(tid, now=NOW + 3 * M + timedelta(seconds=30))
    r = next(x for x in s.active(now=NOW + 4 * M) if x["id"] == tid)
    assert r["paused_total_sec"] == 90.0 and r["due"] == NOW + 4 * M + timedelta(seconds=30)
    assert s.resume(tid, now=NOW + 5 * M) is False  # 止めていないものは再開できない


# ── 道具と命令 ─────────────────────────────────────────────────────────────


def test_the_tools_are_five_and_pause_resume_are_offered():
    t, _, _ = _tool()
    names = {d["name"] for d in t.get_tool_definitions()}
    assert {"pause_timer", "resume_timer"} <= names and len(names) == 5


def test_pausing_a_stopwatch_is_refused():
    t, store, _ = _tool()
    asyncio.run(t.call("start_stopwatch", {"label": "ランニング"}))
    text, ok = asyncio.run(t.call("pause_timer", {"id": 1}))
    assert not ok and "ストップウォッチ" in text


def test_the_slash_commands_pause_and_resume_without_the_llm():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("止めておく：id=1 「パスタ」 残り 2:00", True))
    a._handle_timer_command = lambda ui: Agent._handle_timer_command(a, ui)
    asyncio.run(a._handle_timer_command("/timer pause"))
    a._timer_tool.call.assert_awaited_with("pause_timer", {"id": "all"})
    asyncio.run(a._handle_timer_command("/timer resume 1"))
    a._timer_tool.call.assert_awaited_with("resume_timer", {"id": "1"})


def test_fire_due_skips_a_paused_timer():
    store = MagicMock()
    store.due_now = MagicMock(return_value=[])  # 器の側で除く（SQL）。ここは呼び方の証拠だけ
    dif = MagicMock()
    assert timer_watch.fire_due(store, dif, now=NOW) == 0


def test_pause_resume_through_the_tool_keeps_the_remaining_time():
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    text, ok = asyncio.run(t.call("pause_timer", {"id": 1}))
    assert ok and "止めておく" in text and "残り 3:00" in text
    text, ok = asyncio.run(t.call("resume_timer", {"id": "all"}))
    assert ok and "再開した" in text
    text, ok = asyncio.run(t.call("resume_timer", {"id": 1}))
    assert not ok and "止めていない" in text
