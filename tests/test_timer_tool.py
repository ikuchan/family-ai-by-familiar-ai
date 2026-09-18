"""タイマーの道具（`tools/timer.py`・知-n）。主LLM が呼ぶ 3 本：`set_timer`・`start_stopwatch`・`cancel_timer`。

- 静穏時間に掛かる／黙っているよう頼まれているときは**登録せず**「確かめて」を返す。登録は機械の
  `confirm`（`call(..., confirmed=True)`・出-y）。
- 登録したら O に `予定` の記録、止めたら「やめた」の記録。
- 同時に 1 本（2026-09-18・以前は 5 本〔仮〕）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from familiar_agent.routines import QuietHoursRule
from familiar_agent.tools.timer import MAX_ACTIVE, TimerTool

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 9, 15, 19, 30, tzinfo=JST)


class _FakeStore:
    def __init__(self):
        self.rows: list[dict] = []
        self._next = 1

    def add(self, *, label, due, asked_by, obs_id, passes_quiet, now=None):
        tid = self._next
        self._next += 1
        self.rows.append(
            {
                "id": tid,
                "label": label,
                "due": due,
                "started_at": now or NOW,
                "fired_at": None,
                "cancelled_at": None,
                "asked_by": asked_by,
                "obs_id": obs_id,
                "passes_quiet": passes_quiet,
                "paused_at": None,
                "paused_total_sec": 0.0,
                "listen": False,
            }
        )
        return tid

    def pause(self, tid, *, now=None):
        for r in self.active():
            if r["id"] == tid and r["due"] is not None and r["paused_at"] is None:
                r["paused_at"] = now or NOW
                return True
        return False

    def resume(self, tid, *, now=None):
        for r in self.active():
            if r["id"] == tid and r["paused_at"] is not None:
                gap = (now or NOW) - r["paused_at"]
                r["paused_total_sec"] += gap.total_seconds()
                r["due"] = r["due"] + gap
                r["paused_at"] = None
                return True
        return False

    def active(self, *, now=None):
        return [r for r in self.rows if r["fired_at"] is None and r["cancelled_at"] is None]

    def recently_fired(self, *, now=None, within_sec):
        return [r for r in self.rows if r["fired_at"] is not None]

    def recently_stopped(self, *, now=None, within_sec):
        return [r for r in self.rows if r["cancelled_at"] is not None]

    def set_listen(self, tid, value):
        for r in self.active():
            if r["id"] == tid:
                r["listen"] = bool(value)
                return True
        return False

    def cancel(self, tid, *, now=None):
        for r in self.rows:
            if r["id"] == tid and r["fired_at"] is None and r["cancelled_at"] is None:
                r["cancelled_at"] = now or NOW
                return True
        return False

    def cancel_all(self, *, now=None):
        n = 0
        for r in self.active():
            r["cancelled_at"] = now or NOW
            n += 1
        return n


@pytest.fixture(autouse=True)
def _confirm_off(monkeypatch):
    """ここは掛かったあとの振る舞いを見る。確かめる（`TIMER_CONFIRM`）は `test_timer_confirm_and_single` が見る。"""
    monkeypatch.setenv("TIMER_CONFIRM", "false")


def _tool(*, quiet=QuietHoursRule(23, 7), silence=False):
    store = _FakeStore()
    oif = MagicMock()
    oif.write = AsyncMock(side_effect=lambda mi, **kw: f"obs-{mi.content[:4]}")
    t = TimerTool(
        store=lambda: store,
        oif=oif,
        speaker=lambda: "パパ",
        quiet=lambda: quiet,
        silence_active=lambda: silence,
        now=lambda: NOW,
    )
    return t, store, oif


def test_the_definitions_are_four_tools():
    t, _, _ = _tool()
    names = [d["name"] for d in t.get_tool_definitions()]
    assert names == [
        "set_timer",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
    ]  # ストップウォッチは別物（知-u）
    props = t.get_tool_definitions()[0]["input_schema"]["properties"]
    assert (
        props.keys() == {"after_minutes", "label"} and "at" not in props
    )  # 何時に、はアラーム。`confirmed` は LLM の引数に無い（出-y）


def test_a_daytime_timer_is_registered_at_once_and_written_to_o():
    t, store, oif = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and "id=1" in text and "19:33" in text
    r = store.active()[0]
    assert (
        r["label"] == "パスタ"
        and r["due"] == NOW + timedelta(minutes=3)
        and r["asked_by"] == "パパ"
    )
    assert r["passes_quiet"] is False and r["obs_id"] == "obs-パパに頼"
    mi = oif.write.call_args.args[0]
    assert mi.direction == "予定" and "パスタ" in mi.content and "19:33" in mi.content
    assert oif.write.call_args.kwargs.get(
        "writer_id"
    )  # 書き手を添える（無いと OIF.write が落ちる）


def test_a_quiet_hours_timer_asks_first_then_registers_when_confirmed():
    from datetime import datetime as _dt

    t, store, oif = _tool()
    # 静穏時間は 23〜7。いま 22:58 なら 5 分後の 23:03 は中（何時に、はアラームなので分数で入れる）。
    late = NOW.replace(hour=22, minute=58)
    t._now = lambda: late
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "お茶"}))
    assert ok and "確かめ" in text and "静穏" in text and "23:03" in text
    assert store.active() == [] and not oif.write.called
    text, ok = asyncio.run(
        t.call("set_timer", {"after_minutes": 5, "label": "お茶"}, confirmed=True)
    )
    assert ok and "id=1" in text
    assert store.active()[0]["passes_quiet"] is True
    assert isinstance(late, _dt)


def test_a_silence_request_also_asks_first():
    t, store, _ = _tool(silence=True)
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "x"}))
    assert ok and "確かめ" in text and "黙って" in text and store.active() == []


def test_bad_time_is_refused():
    t, store, _ = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": -1, "label": "x"}))
    assert not ok and "分数" in text and store.active() == []
    text, ok = asyncio.run(t.call("set_timer", {"at": "7:00", "label": "x"}))  # 何時に、はアラーム
    assert not ok and "アラーム" in text and store.active() == []


def test_the_limit_is_one():
    """同時に 1 本（2026-09-18・以前は 5 本〔仮〕）。中身は `test_timer_confirm_and_single`。"""
    assert MAX_ACTIVE == 1
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 1, "label": "t0"}))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 9, "label": "多すぎ"}))
    assert not ok and "t0" in text and len(store.active()) == 1


def test_cancel_one_or_all_and_says_when_nothing_is_running():
    t, store, oif = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "a"}))
    text, ok = asyncio.run(t.call("cancel_timer", {"id": 1}))
    assert ok and "a" in text and "止めた" in text and store.active() == []
    assert oif.write.call_args.args[0].content.startswith("やめた：")
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "b"}))
    text, ok = asyncio.run(t.call("cancel_timer", {"id": "all"}))
    assert ok and "1 本" in text and store.active() == []
    text, ok = asyncio.run(t.call("cancel_timer", {"id": "all"}))
    assert ok and "動いているタイマーは無い" in text
    text, ok = asyncio.run(t.call("cancel_timer", {"id": 99}))
    assert not ok


def test_the_frame_comes_from_the_store():
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 2, "label": "パスタ"}))
    frame = t.frame()
    assert frame.startswith("[タイマー]") and "パスタ" in frame and "残り 2:00" in frame


def test_the_clock_starts_when_the_person_spoke_not_when_the_tool_ran():
    """「はい」「始めて」と言った時刻を起点にする（道具まで 3 秒かかっても起点は発話の瞬間・2026-09-15）。"""
    t, store, _ = _tool()
    said_at = NOW - timedelta(seconds=3)
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 1, "label": "x"}, now=said_at))
    assert ok and store.active()[0]["due"] == said_at + timedelta(minutes=1)
    assert store.active()[0]["started_at"] == said_at


def test_stopping_a_timer_reports_the_remaining_time():
    t, store, _ = _tool()
    asyncio.run(
        t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}, now=NOW - timedelta(minutes=1))
    )
    text, ok = asyncio.run(t.call("cancel_timer", {"id": 1}))
    assert ok and "残り 2:00" in text
