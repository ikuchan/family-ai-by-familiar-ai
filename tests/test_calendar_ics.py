"""ICS を読んで、JST の日付範囲の予定に展開する（知-j・依存ゼロ）。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))

from calendar_mcp.ics import events_between, parse_ics  # noqa: E402

_ICS = """BEGIN:VCALENDAR
VERSION:2.0
X-WR-CALNAME:ファミリー
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260914
DTEND;VALUE=DATE:20260915
SUMMARY:運動会
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Tokyo:20260914T153000
DTEND;TZID=Asia/Tokyo:20260914T170000
SUMMARY:テニス
END:VEVENT
BEGIN:VEVENT
DTSTART:20260913T010000Z
DTEND:20260913T020000Z
SUMMARY:UTC の予定
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Tokyo:20260907T190000
DTEND;TZID=Asia/Tokyo:20260907T200000
RRULE:FREQ=WEEKLY;BYDAY=MO;UNTIL=20261001T000000Z
EXDATE;TZID=Asia/Tokyo:20260921T190000
SUMMARY:サッカー教室
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260901
RRULE:FREQ=MONTHLY;COUNT=3
SUMMARY:給料日
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Tokyo:20260913T090000
DTEND;TZID=Asia/Tokyo:20260913T100000
RRULE:FREQ=HOURLY;COUNT=3
SUMMARY:展開できない繰り返し
END:VEVENT
END:VCALENDAR
"""


def _titles(day: date, days: int = 1):
    cal = parse_ics(_ICS)
    return [
        (e.start_local.strftime("%m-%d %H:%M"), e.all_day, e.summary)
        for e in events_between(cal, day, days)
    ]


def test_the_calendar_name_is_read() -> None:
    assert parse_ics(_ICS).name == "ファミリー"


def test_all_day_and_timed_events_on_the_day() -> None:
    got = _titles(date(2026, 9, 14))
    assert ("09-14 00:00", True, "運動会") in got
    assert ("09-14 15:30", False, "テニス") in got
    assert ("09-14 19:00", False, "サッカー教室") in got  # 月曜の週次


def test_utc_times_are_shown_in_jst() -> None:
    got = _titles(date(2026, 9, 13))
    assert ("09-13 10:00", False, "UTC の予定") in got


def test_weekly_rule_respects_exdate_and_until() -> None:
    assert not any(t == "サッカー教室" for _, _, t in _titles(date(2026, 9, 21)))  # EXDATE
    assert any(t == "サッカー教室" for _, _, t in _titles(date(2026, 9, 28)))
    assert not any(t == "サッカー教室" for _, _, t in _titles(date(2026, 10, 5)))  # UNTIL 後


def test_monthly_count_rule() -> None:
    assert any(t == "給料日" for _, _, t in _titles(date(2026, 11, 1)))
    assert not any(t == "給料日" for _, _, t in _titles(date(2026, 12, 1)))  # COUNT=3 を超えた


def test_an_unsupported_rule_is_not_silently_dropped() -> None:
    cal = parse_ics(_ICS)
    got = [
        e for e in events_between(cal, date(2026, 9, 13), 1) if e.summary == "展開できない繰り返し"
    ]
    assert len(got) == 1 and got[0].note == "繰り返し（展開できない形）"


def test_a_range_of_days_is_sorted() -> None:
    got = _titles(date(2026, 9, 13), 2)
    assert [t for _, _, t in got][:3] == [
        "展開できない繰り返し",
        "UTC の予定",
        "運動会",
    ]  # 09:00 → 10:00 → 翌日の終日


# ── 情-s（2026-10-17）：ICS の VEVENT の順が変わっても、並びは変わらない ──────────────


def _ics(*events: str) -> str:
    return "BEGIN:VCALENDAR\nVERSION:2.0\n" + "".join(events) + "END:VCALENDAR\n"


_ALL_DAY_B = "BEGIN:VEVENT\nDTSTART;VALUE=DATE:20261018\nSUMMARY:遠足\nEND:VEVENT\n"
_ALL_DAY_A = "BEGIN:VEVENT\nDTSTART;VALUE=DATE:20261018\nSUMMARY:お弁当\nEND:VEVENT\n"
_AT_FOUR_B = (
    "BEGIN:VEVENT\nDTSTART;TZID=Asia/Tokyo:20261018T160000\n"
    "DTEND;TZID=Asia/Tokyo:20261018T180000\nSUMMARY:塾\nEND:VEVENT\n"
)
_AT_FOUR_A = (
    "BEGIN:VEVENT\nDTSTART;TZID=Asia/Tokyo:20261018T160000\n"
    "DTEND;TZID=Asia/Tokyo:20261018T170000\nSUMMARY:塾\nEND:VEVENT\n"
)


def _order(text: str):
    return [
        (e.summary, e.end_local) for e in events_between(parse_ics(text), date(2026, 10, 18), 1)
    ]


def test_all_day_events_on_the_same_day_keep_one_order() -> None:
    assert _order(_ics(_ALL_DAY_A, _ALL_DAY_B)) == _order(_ics(_ALL_DAY_B, _ALL_DAY_A))


def test_events_starting_together_keep_one_order() -> None:
    """同じ名前・同じ開始でも、終わりの時刻で決まった順になる。"""
    assert _order(_ics(_AT_FOUR_A, _AT_FOUR_B)) == _order(_ics(_AT_FOUR_B, _AT_FOUR_A))
