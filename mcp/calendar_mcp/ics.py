"""ICS（iCalendar）を読み、JST の日付範囲の予定へ展開する（依存ゼロ）。

読むのは `VEVENT` の `DTSTART`／`DTEND`／`SUMMARY`／`RRULE`／`EXDATE` だけ。繰り返しは
`FREQ=DAILY|WEEKLY|MONTHLY|YEARLY` と `UNTIL`／`COUNT`／`BYDAY`／`INTERVAL` まで展開する。
それ以外の繰り返しは**黙って落とさず**、その日の予定として題名だけ出し「展開できない形」と添える。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Tokyo")
_WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
_SUPPORTED_FREQ = ("DAILY", "WEEKLY", "MONTHLY", "YEARLY")


@dataclass(frozen=True)
class Event:
    summary: str
    start_local: datetime  # JST。終日は 00:00
    end_local: datetime | None
    all_day: bool
    note: str = ""  # 「繰り返し（展開できない形）」など


@dataclass(frozen=True)
class RawEvent:
    summary: str
    start: datetime | date
    end: datetime | date | None
    rrule: dict[str, str] | None
    exdates: tuple[date, ...]


@dataclass(frozen=True)
class Calendar:
    name: str
    events: tuple[RawEvent, ...]


def _unfold(text: str) -> list[str]:
    """折り返し（次行が空白で始まる）を戻す。"""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_dt(value: str, params: dict[str, str]) -> datetime | date:
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    m = re.fullmatch(r"(\d{8})T(\d{6})(Z?)", value)
    if not m:
        raise ValueError(f"日時が読めない: {value}")
    d, t, z = m.groups()
    naive = datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(t[:2]), int(t[2:4]), int(t[4:6]))
    if z:
        return naive.replace(tzinfo=timezone.utc).astimezone(TZ)
    tzid = params.get("TZID")
    tz = ZoneInfo(tzid) if tzid else TZ
    return naive.replace(tzinfo=tz).astimezone(TZ)


def _split_prop(line: str) -> tuple[str, dict[str, str], str]:
    head, _, value = line.partition(":")
    parts = head.split(";")
    name = parts[0].upper()
    params = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        params[k.upper()] = v
    return name, params, value


def parse_ics(text: str) -> Calendar:
    name = ""
    events: list[RawEvent] = []
    cur: dict | None = None
    for line in _unfold(text):
        if not line:
            continue
        pname, params, value = _split_prop(line)
        if pname == "X-WR-CALNAME":
            name = value.strip()
        elif pname == "BEGIN" and value.strip() == "VEVENT":
            cur = {"summary": "", "start": None, "end": None, "rrule": None, "exdates": []}
        elif pname == "END" and value.strip() == "VEVENT" and cur is not None:
            if cur["start"] is not None:
                events.append(
                    RawEvent(
                        summary=cur["summary"],
                        start=cur["start"],
                        end=cur["end"],
                        rrule=cur["rrule"],
                        exdates=tuple(cur["exdates"]),
                    )
                )
            cur = None
        elif cur is not None:
            if pname == "SUMMARY":
                cur["summary"] = value.replace("\\,", ",").replace("\\n", " ").strip()
            elif pname == "DTSTART":
                cur["start"] = _parse_dt(value, params)
            elif pname == "DTEND":
                cur["end"] = _parse_dt(value, params)
            elif pname == "RRULE":
                cur["rrule"] = {
                    k.upper(): v for k, _, v in (p.partition("=") for p in value.split(";"))
                }
            elif pname == "EXDATE":
                for v in value.split(","):
                    d = _parse_dt(v, params)
                    cur["exdates"].append(
                        d if isinstance(d, date) and not isinstance(d, datetime) else d.date()
                    )
    return Calendar(name=name, events=tuple(events))


def _local_start(start: datetime | date) -> tuple[datetime, bool]:
    if isinstance(start, datetime):
        return start.astimezone(TZ), False
    return datetime.combine(start, time(0, 0), tzinfo=TZ), True


def _add_months(d: date, n: int) -> date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(
        d.day,
        [
            31,
            29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ][month - 1],
    )
    return date(year, month, day)


def _until(rule: dict[str, str]) -> date | None:
    v = rule.get("UNTIL")
    if not v:
        return None
    d = _parse_dt(v, {})
    return d.date() if isinstance(d, datetime) else d


def _occurrences(ev: RawEvent, first: date, last: date) -> list[tuple[date, str]]:
    """その予定が `first`〜`last`（両端含む）に起きる日と但し書きを返す。"""
    start_local, _ = _local_start(ev.start)
    base = start_local.date()
    if not ev.rrule:
        return [(base, "")] if first <= base <= last else []
    freq = ev.rrule.get("FREQ", "")
    if freq not in _SUPPORTED_FREQ:
        # 展開できない形。黙って落とさず、基準日が範囲内ならそのまま出す。
        return [(base, "繰り返し（展開できない形）")] if first <= base <= last else []
    interval = max(1, int(ev.rrule.get("INTERVAL", "1")))
    count = int(ev.rrule["COUNT"]) if "COUNT" in ev.rrule else None
    until = _until(ev.rrule)
    bydays = [
        _WEEKDAYS[x[-2:]] for x in ev.rrule.get("BYDAY", "").split(",") if x[-2:] in _WEEKDAYS
    ]
    out: list[tuple[date, str]] = []
    n = 0
    step = 0
    while True:
        if freq == "DAILY":
            d = base + timedelta(days=step * interval)
            cands = [d]
        elif freq == "WEEKLY":
            week_start = base - timedelta(days=base.weekday()) + timedelta(weeks=step * interval)
            days = bydays or [base.weekday()]
            cands = [week_start + timedelta(days=w) for w in sorted(days)]
            cands = [c for c in cands if c >= base]
        elif freq == "MONTHLY":
            cands = [_add_months(base, step * interval)]
        else:  # YEARLY
            cands = [_add_months(base, 12 * step * interval)]
        for d in cands:
            if until and d > until:
                return out
            n += 1
            if count is not None and n > count:
                return out
            if d in ev.exdates:
                continue
            if d > last:
                return out
            if d >= first:
                out.append((d, ""))
        step += 1
        if step > 10000:
            return out


def events_between(cal: Calendar, first: date, days: int) -> list[Event]:
    last = first + timedelta(days=max(1, days) - 1)
    found: list[Event] = []
    for ev in cal.events:
        start_local, all_day = _local_start(ev.start)
        duration = None
        if ev.end is not None:
            end_local, _ = _local_start(ev.end)
            duration = end_local - start_local
        for d, note in _occurrences(ev, first, last):
            s = (
                datetime.combine(d, start_local.timetz())
                if not all_day
                else datetime.combine(d, time(0, 0), tzinfo=TZ)
            )
            e = (s + duration) if duration is not None else None
            found.append(
                Event(summary=ev.summary, start_local=s, end_local=e, all_day=all_day, note=note)
            )
    found.sort(key=lambda e: (e.start_local.date(), not e.all_day, e.start_local))
    return found
