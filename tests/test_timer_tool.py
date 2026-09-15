"""タイマーの道具（`tools/timer.py`・知-n）。主LLM が呼ぶ 3 本：`set_timer`・`start_stopwatch`・`cancel_timer`。

- 静穏時間に掛かる／黙っているよう頼まれているときは**登録せず**「確かめて」を返す。`confirmed` で登録。
- 登録したら O に `予定` の記録、止めたら「やめた」の記録。
- 同時に 5 本〔仮〕まで。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
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
            }
        )
        return tid

    def active(self, *, now=None):
        return [r for r in self.rows if r["fired_at"] is None and r["cancelled_at"] is None]

    def recently_fired(self, *, now=None, within_sec):
        return [r for r in self.rows if r["fired_at"] is not None]

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


def test_the_definitions_are_three_tools():
    t, _, _ = _tool()
    names = [d["name"] for d in t.get_tool_definitions()]
    assert names == ["set_timer", "start_stopwatch", "cancel_timer"]
    assert t.get_tool_definitions()[0]["input_schema"]["properties"].keys() >= {
        "after_minutes",
        "at",
        "label",
        "confirmed",
    }


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


def test_a_quiet_hours_timer_asks_first_then_registers_when_confirmed():
    t, store, oif = _tool()
    # 静穏時間は 23〜7（7:00 は端の外）。6:30 は中。
    text, ok = asyncio.run(t.call("set_timer", {"at": "6:30", "label": "起こす"}))
    assert ok and "確かめ" in text and "静穏" in text and "06:30" in text
    assert store.active() == [] and not oif.write.called
    text, ok = asyncio.run(
        t.call("set_timer", {"at": "6:30", "label": "起こす", "confirmed": True})
    )
    assert ok and "id=1" in text
    assert store.active()[0]["passes_quiet"] is True


def test_a_silence_request_also_asks_first():
    t, store, _ = _tool(silence=True)
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "x"}))
    assert ok and "確かめ" in text and "黙って" in text and store.active() == []


def test_bad_time_is_refused():
    t, store, _ = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"at": "あした", "label": "x"}))
    assert not ok and "読めない" in text and store.active() == []


def test_the_limit_is_five():
    t, store, _ = _tool()
    for i in range(MAX_ACTIVE):
        asyncio.run(t.call("set_timer", {"after_minutes": 1 + i, "label": f"t{i}"}))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 9, "label": "多すぎ"}))
    assert not ok and f"{MAX_ACTIVE}" in text and len(store.active()) == MAX_ACTIVE


def test_stopwatch_starts_without_due():
    t, store, oif = _tool()
    text, ok = asyncio.run(t.call("start_stopwatch", {"label": "ランニング"}))
    assert ok and "id=1" in text and store.active()[0]["due"] is None
    assert oif.write.call_args.args[0].direction == "予定"


def test_cancel_one_or_all_and_says_when_nothing_is_running():
    t, store, oif = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "a"}))
    asyncio.run(t.call("set_timer", {"after_minutes": 4, "label": "b"}))
    text, ok = asyncio.run(t.call("cancel_timer", {"id": 1}))
    assert ok and "a" in text and "止めた" in text and len(store.active()) == 1
    assert oif.write.call_args.args[0].content.startswith("やめた：")
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
