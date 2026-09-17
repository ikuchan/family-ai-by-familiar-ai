"""アラームはタイマーと別物（知-q・2026-09-18・`設計方針_アラーム` v0.1）。

遠い時刻に起こす・知らせる。それまでは普通に暮らす——黙らない・聞かない状態にしない・確認は静穏時間に
鳴るときだけ・同時 5 本〔仮〕・一時停止なし。共有するのは基盤だけ（音の再生・T の tick・`予定` の記録・
静穏時間の判定）。時刻の読み（「7 時半」）はここに持ち、タイマーは分数だけ。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.core import alarm_rules, timer_rules
from familiar_agent.loop import alarm_watch
from familiar_agent.routines import QuietHoursRule
from familiar_agent.store.alarms import AlarmStore
from familiar_agent.tools.alarm import MAX_ACTIVE, AlarmTool

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 9, 18, 10, 0, tzinfo=JST)


# ── 規則 ─────────────────────────────────────────────────────────────────


def test_resolve_at_reads_a_local_time_and_rolls_to_tomorrow_when_past():
    assert alarm_rules.resolve_at("21:00", now=NOW) == NOW.replace(hour=21, minute=0)
    assert alarm_rules.resolve_at("7:00", now=NOW) == (NOW + timedelta(days=1)).replace(
        hour=7, minute=0
    )
    assert alarm_rules.resolve_at("７時半", now=NOW) == (NOW + timedelta(days=1)).replace(
        hour=7, minute=30
    )
    with pytest.raises(ValueError):
        alarm_rules.resolve_at("あした", now=NOW)


def test_timer_rules_no_longer_read_a_clock_time():
    assert not hasattr(timer_rules, "_AT_RE")
    with pytest.raises(TypeError):
        timer_rules.resolve_due(after_minutes=None, at="7:00", now=NOW)  # type: ignore[call-arg]


def test_confirmation_is_only_about_quiet_hours():
    quiet = QuietHoursRule(23, 7)
    assert alarm_rules.needs_confirmation(NOW.replace(hour=6, minute=30), quiet=quiet)
    assert alarm_rules.needs_confirmation(NOW.replace(hour=12), quiet=quiet) is None


def test_the_frame_lists_the_next_alarms():
    rows = [{"id": 1, "label": "起こす", "at": NOW.replace(hour=6, minute=30) + timedelta(days=1)}]
    text = alarm_rules.render_frame(rows, [], now=NOW)
    assert text.startswith("[アラーム]") and "06:30" in text and "起こす" in text


# ── 器（実 DB）────────────────────────────────────────────────────────────


def _store() -> AlarmStore:
    conn = psycopg2.connect(
        os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor
    )
    conn.autocommit = True
    return AlarmStore(conn)


def test_the_store_adds_lists_fires_and_cancels():
    s = _store()
    aid = s.add(
        label="起こす",
        at=NOW + timedelta(hours=1),
        asked_by="パパ",
        obs_id=None,
        passes_quiet=False,
        now=NOW,
    )
    assert [r["id"] for r in s.active(now=NOW)][-1] == aid
    assert s.due_now(now=NOW) == [] or all(r["id"] != aid for r in s.due_now(now=NOW))
    assert [r["id"] for r in s.due_now(now=NOW + timedelta(hours=2))][-1] == aid
    assert s.mark_fired(aid, now=NOW + timedelta(hours=2)) and not s.mark_fired(aid, now=NOW)
    bid = s.add(
        label="薬",
        at=NOW + timedelta(hours=3),
        asked_by="",
        obs_id=None,
        passes_quiet=True,
        now=NOW,
    )
    assert s.cancel(bid, now=NOW) and not s.cancel(bid, now=NOW)


# ── 道具 ─────────────────────────────────────────────────────────────────


class _FakeStore:
    def __init__(self):
        self.rows: list[dict] = []
        self._next = 1

    def add(self, *, label, at, asked_by, obs_id, passes_quiet, now=None):
        aid = self._next
        self._next += 1
        self.rows.append(
            {
                "id": aid,
                "label": label,
                "at": at,
                "set_at": now or NOW,
                "fired_at": None,
                "cancelled_at": None,
                "asked_by": asked_by,
                "obs_id": obs_id,
                "passes_quiet": passes_quiet,
            }
        )
        return aid

    def active(self, *, now=None):
        return [r for r in self.rows if r["fired_at"] is None and r["cancelled_at"] is None]

    def recently_fired(self, *, now=None, within_sec):
        return [r for r in self.rows if r["fired_at"] is not None]

    def cancel(self, aid, *, now=None):
        for r in self.active():
            if r["id"] == aid:
                r["cancelled_at"] = now or NOW
                return True
        return False

    def cancel_all(self, *, now=None):
        n = 0
        for r in self.active():
            r["cancelled_at"] = now or NOW
            n += 1
        return n


def _tool(*, quiet=QuietHoursRule(23, 7)):
    store = _FakeStore()
    oif = MagicMock()
    oif.write = AsyncMock(return_value="obs-1")
    on_cancel = MagicMock()
    t = AlarmTool(
        store=lambda: store,
        oif=oif,
        speaker=lambda: "パパ",
        quiet=lambda: quiet,
        now=lambda: NOW,
        on_cancel=on_cancel,
    )
    return t, store, oif, on_cancel


def test_the_tools_are_set_and_cancel_only():
    t, _, _, _ = _tool()
    names = [d["name"] for d in t.get_tool_definitions()]
    assert names == ["set_alarm", "cancel_alarm"]
    props = t.get_tool_definitions()[0]["input_schema"]["properties"]
    assert "at" in props and "after_minutes" not in props


def test_a_daytime_alarm_is_set_at_once_and_written_to_o(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "true")  # タイマーの設定はアラームに効かない
    monkeypatch.setenv("TIMER_SILENCE", "true")
    t, store, oif, _ = _tool()
    text, ok = asyncio.run(t.call("set_alarm", {"at": "21:00", "label": "薬"}))
    assert ok and "id=1" in text and "21:00" in text and len(store.active()) == 1
    assert "アラーム" in oif.write.call_args.args[0].content


def test_a_quiet_hours_alarm_asks_first_then_sets_when_confirmed():
    t, store, _, _ = _tool()
    text, ok = asyncio.run(t.call("set_alarm", {"at": "6:30", "label": "起こす"}))
    assert ok and "確かめて" in text and store.active() == []
    text, ok = asyncio.run(
        t.call("set_alarm", {"at": "6:30", "label": "起こす", "confirmed": True})
    )
    assert ok and store.active()[0]["passes_quiet"] is True


def test_up_to_five_alarms():
    t, store, _, _ = _tool()
    for i in range(MAX_ACTIVE):
        asyncio.run(t.call("set_alarm", {"at": f"{12 + i}:00", "label": f"a{i}"}))
    text, ok = asyncio.run(t.call("set_alarm", {"at": "20:00", "label": "多すぎ"}))
    assert not ok and f"{MAX_ACTIVE}" in text and len(store.active()) == MAX_ACTIVE


def test_cancel_one_or_all_and_the_ring_stops_first():
    t, store, _, on_cancel = _tool()
    asyncio.run(t.call("set_alarm", {"at": "21:00", "label": "薬"}))
    asyncio.run(t.call("set_alarm", {"at": "22:00", "label": "戸締り"}))
    text, ok = asyncio.run(t.call("cancel_alarm", {"id": 1}))
    assert ok and "薬" in text and len(store.active()) == 1 and on_cancel.called
    text, ok = asyncio.run(t.call("cancel_alarm", {"id": "all"}))
    assert ok and store.active() == []


# ── 鳴らす ───────────────────────────────────────────────────────────────


def test_due_alarms_ring_and_push_a_device_request():
    store = MagicMock()
    store.due_now = MagicMock(
        return_value=[
            {
                "id": 1,
                "label": "起こす",
                "at": NOW - timedelta(seconds=5),
                "asked_by": "パパ",
                "passes_quiet": True,
            }
        ]
    )
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    assert alarm_watch.fire_due(store, dif, now=NOW, ring_sec=30.0, quiet=True, gain=1.5) == 1
    dif.ring.assert_called_once_with(seconds=30.0, gain=1.5)
    kind, content = dif.device.call_args.args[0], dif.device.call_args.args[1]
    assert (
        kind == "アラーム"
        and "起こす" in content
        and dif.device.call_args.kwargs["passes_gate"] is True
    )


def test_an_unconfirmed_alarm_in_quiet_hours_makes_no_sound():
    store = MagicMock()
    store.due_now = MagicMock(
        return_value=[{"id": 1, "label": "薬", "at": NOW, "asked_by": "", "passes_quiet": False}]
    )
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    alarm_watch.fire_due(store, dif, now=NOW, ring_sec=30.0, quiet=True)
    dif.ring.assert_not_called()


def test_the_slash_command_stops_an_alarm_without_the_llm():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._alarm_tool = MagicMock()
    a._alarm_tool.call = AsyncMock(return_value=("止めた：id=2 「薬」", True))
    a._handle_alarm_command = lambda ui: Agent._handle_alarm_command(a, ui)
    assert asyncio.run(a._handle_alarm_command("/alarm stop 2")) == "止めた：id=2 「薬」"
    a._alarm_tool.call.assert_awaited_once_with("cancel_alarm", {"id": "2"})
    assert asyncio.run(a._handle_alarm_command("/timer stop")) is None
