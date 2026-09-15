"""タイマーの規則（`core/timer_rules.py`・純関数・知-n）。

- `resolve_due`：「3 分後」か「7:00」（JST・過ぎていれば翌日）を絶対時刻に。
- `needs_confirmation`：鳴る時刻が静穏時間に入る／沈黙の依頼が生きているなら、登録せず一度確かめる。
- `render_frame`：`[タイマー]` の枠（残り／経過・直前に鳴ったもの）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from familiar_agent.core import timer_rules as tr
from familiar_agent.routines import QuietHoursRule

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 9, 15, 19, 30, tzinfo=JST)  # 19:30 JST


def test_after_minutes_is_added_to_now():
    assert tr.resolve_due(after_minutes=3, at=None, now=NOW) == NOW + timedelta(minutes=3)
    assert tr.resolve_due(after_minutes=0.5, at=None, now=NOW) == NOW + timedelta(seconds=30)


def test_at_is_today_if_ahead_else_tomorrow():
    assert tr.resolve_due(after_minutes=None, at="21:00", now=NOW) == NOW.replace(
        hour=21, minute=0, second=0, microsecond=0
    )
    assert tr.resolve_due(after_minutes=None, at="7:00", now=NOW) == (
        NOW + timedelta(days=1)
    ).replace(hour=7, minute=0, second=0, microsecond=0)
    assert tr.resolve_due(after_minutes=None, at="７時半", now=NOW) == (
        NOW + timedelta(days=1)
    ).replace(hour=7, minute=30, second=0, microsecond=0)


def test_bad_inputs_are_refused_with_a_reason():
    for kw in (
        dict(after_minutes=None, at=None),
        dict(after_minutes=-1, at=None),
        dict(after_minutes=None, at="あした"),
        dict(after_minutes=3, at="7:00"),
    ):
        try:
            tr.resolve_due(now=NOW, **kw)
        except ValueError as e:
            assert str(e)
        else:
            raise AssertionError(kw)


def test_confirmation_is_needed_when_the_due_falls_in_quiet_hours_or_a_silence_request_is_alive():
    quiet = QuietHoursRule(start_hour=23, end_hour=7)
    due_day = NOW.replace(hour=21)
    due_night = NOW.replace(hour=23, minute=30)
    due_dawn = (NOW + timedelta(days=1)).replace(hour=6, minute=59)
    assert tr.needs_confirmation(due_day, quiet=quiet, silence_active=False) is None
    assert "静穏" in (tr.needs_confirmation(due_night, quiet=quiet, silence_active=False) or "")
    assert "静穏" in (tr.needs_confirmation(due_dawn, quiet=quiet, silence_active=False) or "")
    assert "黙って" in (tr.needs_confirmation(due_day, quiet=quiet, silence_active=True) or "")
    assert (
        tr.needs_confirmation(None, quiet=quiet, silence_active=True) is None
    )  # ストップウォッチは鳴らない


def test_the_frame_shows_remaining_and_elapsed_and_recently_fired():
    rows = [
        {
            "id": 3,
            "label": "パスタ",
            "due": NOW + timedelta(minutes=2, seconds=5),
            "started_at": NOW - timedelta(minutes=1),
            "fired_at": None,
        },
        {
            "id": 4,
            "label": "ランニング",
            "due": None,
            "started_at": NOW - timedelta(minutes=12, seconds=30),
            "fired_at": None,
        },
    ]
    fired = [
        {
            "id": 2,
            "label": "お茶",
            "due": NOW - timedelta(minutes=1),
            "started_at": NOW - timedelta(minutes=4),
            "fired_at": NOW - timedelta(minutes=1),
        }
    ]
    text = tr.render_frame(rows, fired, now=NOW)
    assert text.startswith("[タイマー]")
    assert "id=3 パスタ 残り 2:05（19:32 に鳴る）" in text
    assert "id=4 ランニング 経過 12:30" in text
    assert "id=2 お茶 は 1 分前に鳴った（もう止まっている）" in text
    assert tr.render_frame([], [], now=NOW) == ""
