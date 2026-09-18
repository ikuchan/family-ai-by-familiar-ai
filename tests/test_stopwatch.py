"""ストップウォッチはタイマーとは別物（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。

09-16 に主LLM が始めた 2 本が 2 日間動き続け、`/timer stop` の返事「2 本止めた：「何のために時間を測るか不明」…」で
何を止めたか分からなかった。表・器・道具・命令・枠・T の見張りを分ける（知-q のアラームと同じ形）：
表 `stopwatches`（068）・`start_stopwatch`／`stop_stopwatch`・`/stopwatch stop`・`[ストップウォッチ]`・
寿命 `STOPWATCH_MAX_SEC`（6 時間）で自動停止。タイマーの道具・命令・表からストップウォッチは消える。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from familiar_agent.core import stopwatch_rules
from familiar_agent.loop import stopwatch_watch
from familiar_agent.store.stopwatches import StopwatchStore
from familiar_agent.tools.stopwatch import StopwatchTool

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
SIX_H = 6 * 3600.0


# ── 規則 ──────────────────────────────────────────────────────────────────


def test_elapsed_text_speaks_like_a_person():
    assert stopwatch_rules.elapsed_text(42) == "42 秒"
    assert stopwatch_rules.elapsed_text(3492) == "58 分 12 秒"
    assert stopwatch_rules.elapsed_text(5 * 3600 + 58 * 60) == "5 時間 58 分"
    assert stopwatch_rules.elapsed_text(2 * 86400 + 3 * 3600) == "2 日 3 時間"


def _row(wid, label, *, started, stopped=None, expired=False):
    return {
        "id": wid,
        "label": label,
        "started_at": started,
        "stopped_at": stopped,
        "asked_by": "パパ",
        "obs_id": None,
        "expired": expired,
    }


def test_the_frame_shows_elapsed_and_warns_before_the_limit():
    running = _row(1, "お風呂", started=NOW - timedelta(hours=5, minutes=50))
    text = stopwatch_rules.render_frame([running], [], now=NOW, max_sec=SIX_H)
    assert (
        text.startswith("[ストップウォッチ]")
        and "5 時間 50 分" in text
        and "あと 10 分で自動で止まる" in text
    )
    fresh = _row(2, "散歩", started=NOW - timedelta(minutes=3))
    assert "自動で止まる" not in stopwatch_rules.render_frame([fresh], [], now=NOW, max_sec=SIX_H)
    stopped = _row(
        3,
        "昼寝",
        started=NOW - timedelta(days=2, hours=3),
        stopped=NOW - timedelta(minutes=1),
        expired=True,
    )
    text = stopwatch_rules.render_frame([], [stopped], now=NOW, max_sec=SIX_H)
    assert "1 分前に寿命で自動で止めた（2 日 2 時間）" in text
    assert stopwatch_rules.render_frame([], [], now=NOW, max_sec=SIX_H) == ""


# ── 器（実 DB）────────────────────────────────────────────────────────────


def _store() -> StopwatchStore:
    conn = psycopg2.connect(
        os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor
    )
    conn.autocommit = True
    return StopwatchStore(conn)


def test_the_store_adds_stops_and_expires():
    s = _store()
    s.stop_all(now=NOW)  # 前のテストの残りを掃く
    wid = s.add(label="お風呂", asked_by="パパ", obs_id=None, now=NOW)
    assert [r["id"] for r in s.active(now=NOW)] == [wid]
    assert s.stop(wid, now=NOW + timedelta(minutes=5)) and not s.stop(wid, now=NOW)
    assert s.active(now=NOW) == []
    got = s.recently_stopped(now=NOW + timedelta(minutes=6), within_sec=180.0)
    mine = next(r for r in got if r["id"] == wid)  # 前の走行の同じ時刻の行が残っていても id で見る
    assert mine["expired"] is False and mine["stopped_at"] == NOW + timedelta(minutes=5)
    old = s.add(label="昼寝", asked_by="", obs_id=None, now=NOW - timedelta(hours=7))
    assert s.expire(now=NOW - timedelta(hours=2), max_sec=SIX_H) == []  # まだ 5 時間
    rows = s.expire(now=NOW, max_sec=SIX_H)
    assert old in [r["id"] for r in rows] and all(r["expired"] for r in rows)
    assert s.active(now=NOW) == []


# ── 道具 ──────────────────────────────────────────────────────────────────


class _FakeStore:
    def __init__(self):
        self.rows: list[dict] = []
        self._next = 1

    def add(self, *, label, asked_by, obs_id, now):
        wid = self._next
        self._next += 1
        self.rows.append(_row(wid, label, started=now))
        return wid

    def active(self, *, now=None):
        return [r for r in self.rows if r["stopped_at"] is None]

    def recently_stopped(self, *, now=None, within_sec):
        return [r for r in self.rows if r["stopped_at"] is not None]

    def stop(self, wid, *, now=None, expired=False):
        for r in self.rows:
            if r["id"] == wid and r["stopped_at"] is None:
                r["stopped_at"], r["expired"] = now, expired
                return True
        return False

    def stop_all(self, *, now=None):
        return sum(self.stop(r["id"], now=now) for r in list(self.rows))

    def expire(self, *, now, max_sec):
        out = []
        for r in self.active():
            if (now - r["started_at"]).total_seconds() >= max_sec:
                self.stop(r["id"], now=now, expired=True)
                out.append(r)
        return out


def _tool():
    store = _FakeStore()
    oif = MagicMock()
    oif.write = AsyncMock(return_value="obs-1")
    t = StopwatchTool(
        store=lambda: store, oif=oif, speaker=lambda: "パパ", max_sec=SIX_H, now=lambda: NOW
    )
    return t, store, oif


def test_the_definitions_are_two_tools():
    t, _, _ = _tool()
    assert [d["name"] for d in t.get_tool_definitions()] == ["start_stopwatch", "stop_stopwatch"]


def test_start_then_stop_reports_the_elapsed_time():
    t, store, oif = _tool()
    text, ok = asyncio.run(t.call("start_stopwatch", {"label": "お風呂"}))
    assert ok and "id=1" in text and "12:00 から" in text
    assert (
        oif.write.call_args.args[0].direction == "予定"
        and "測り始めた" in oif.write.call_args.args[0].content
    )
    text, ok = asyncio.run(t.call("start_stopwatch", {"label": "散歩"}))
    assert not ok and "お風呂" in text and "止めてから" in text  # 同時に 1 本
    text, ok = asyncio.run(
        t.call("stop_stopwatch", {"id": 1}, now=NOW + timedelta(minutes=58, seconds=12))
    )
    assert ok and text == "ストップウォッチ「お風呂」を止めた（58 分 12 秒 経過）"
    text, ok = asyncio.run(t.call("stop_stopwatch", {"id": "all"}))
    assert ok and "無い" in text


def test_the_frame_comes_from_the_tool():
    t, store, _ = _tool()
    asyncio.run(t.call("start_stopwatch", {"label": "お風呂"}))
    assert t.frame().startswith("[ストップウォッチ]") and "お風呂" in t.frame()


# ── T の見張り ─────────────────────────────────────────────────────────────


def test_the_watch_expires_and_writes_a_record():
    store = _FakeStore()
    store.add(label="昼寝", asked_by="", obs_id=None, now=NOW - timedelta(hours=7))
    oif = MagicMock()
    oif.write = AsyncMock(return_value="obs-2")
    n = asyncio.run(stopwatch_watch.expire(store, oif, now=NOW, max_sec=SIX_H))
    assert n == 1 and store.active() == [] and store.rows[0]["expired"] is True
    assert "6 時間で止めた" in oif.write.call_args.args[0].content


# ── タイマーの側からは消える ─────────────────────────────────────────────────


def test_the_timer_tool_no_longer_knows_stopwatches():
    from tests.test_timer_tool import _tool as _timer_tool

    t, _, _ = _timer_tool()
    names = [d["name"] for d in t.get_tool_definitions()]
    assert names == ["set_timer", "cancel_timer", "pause_timer", "resume_timer"]
    text, ok = asyncio.run(t.call("start_stopwatch", {"label": "x"}))
    assert not ok


# ── 命令とループ ───────────────────────────────────────────────────────────


def test_the_slash_command_stops_a_stopwatch_without_the_llm():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._stopwatch_tool = MagicMock()
    a._stopwatch_tool.call = AsyncMock(return_value=("止めた", True))
    assert asyncio.run(Agent._handle_stopwatch_command(a, "/stopwatch stop・")) == "止めた"
    a._stopwatch_tool.call.assert_awaited_with("stop_stopwatch", {"id": "all"})
    assert asyncio.run(Agent._handle_stopwatch_command(a, "/stopwatch stop 3")) == "止めた"
    a._stopwatch_tool.call.assert_awaited_with("stop_stopwatch", {"id": "3"})
    assert asyncio.run(Agent._handle_stopwatch_command(a, "/timer stop")) is None


def test_the_loop_routes_stopwatch_actions_to_the_stopwatch_tool():
    from familiar_agent.loop import workspace
    from familiar_agent.loop.event_loop import (
        _STOPWATCH_ACTIONS,
        _TIMER_ACTIONS,
        InformationProcessing,
        _query_label,
    )

    from tests.test_event_loop import _agent

    assert (
        _STOPWATCH_ACTIONS == ("start_stopwatch", "stop_stopwatch")
        and "start_stopwatch" not in _TIMER_ACTIONS
    )
    assert _query_label("start_stopwatch", {"label": "お風呂"}) == "測り始める「お風呂」"
    assert _query_label("stop_stopwatch", {"id": "all"}) == "測るのを止める「all」"
    assert {"start_stopwatch", "stop_stopwatch"} <= workspace.RETURN_WITHOUT_RECALL
    a = _agent(stream_returns=[])
    a._stopwatch_tool = MagicMock()
    a._stopwatch_tool.call = AsyncMock(return_value=("測り始めた：id=1", True))
    ip = InformationProcessing(a)
    asyncio.run(ip._begin_request(kind="発話", text="今から測って", utterance="今から測って"))
    asyncio.run(
        ip._run_lookup_body("start_stopwatch", {"label": "お風呂"}, "測り始める「お風呂」", None, 1)
    )
    a._stopwatch_tool.call.assert_awaited_once()
    assert a._stopwatch_tool.call.call_args.kwargs["now"] == ip._req.began_at
    assert ip._triggers.get_nowait().result.startswith("測り始めた")


def test_the_arbiter_offers_stop_stopwatch():
    from familiar_agent.loop.arbiter import _EXTRA_ACTIONS

    assert "stop_stopwatch" in _EXTRA_ACTIONS and "start_stopwatch" in _EXTRA_ACTIONS


def test_the_panel_shows_the_stopwatch_frame():
    from familiar_agent.gui import format_timer_rows

    rows = format_timer_rows(
        "",
        ringing=False,
        alarm_frame="",
        stopwatch_frame="[ストップウォッチ]\n- id=1 お風呂 経過 3 分 2 秒（12:00 から）",
    )
    assert rows == ["id=1 お風呂 経過 3 分 2 秒（12:00 から）"]
