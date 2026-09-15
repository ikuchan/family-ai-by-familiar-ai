"""タイマーの器（`store/timers.py`・065・知-n）。状態は列で持ち、再起動をまたいで残る。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

from familiar_agent.store.timers import TimerStore

_DB_URL = os.environ["DATABASE_URL"]
NOW = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)


def _store() -> TimerStore:
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return TimerStore(conn)


def test_add_then_active_lists_it_with_its_columns():
    s = _store()
    tid = s.add(
        label="3 分測る",
        due=NOW + timedelta(minutes=3),
        asked_by="パパ",
        obs_id="o1",
        passes_quiet=False,
        now=NOW,
    )
    rows = s.active(now=NOW)
    assert [r["id"] for r in rows] == [tid]
    r = rows[0]
    assert r["label"] == "3 分測る" and r["asked_by"] == "パパ" and r["obs_id"] == "o1"
    assert r["due"] == NOW + timedelta(minutes=3) and r["started_at"] == NOW
    assert r["fired_at"] is None and r["cancelled_at"] is None and r["passes_quiet"] is False


def test_a_stopwatch_has_no_due_and_never_comes_due():
    s = _store()
    tid = s.add(label="計る", due=None, asked_by="パパ", obs_id=None, passes_quiet=False, now=NOW)
    assert s.due_now(now=NOW + timedelta(days=1)) == []
    assert [r["id"] for r in s.active(now=NOW + timedelta(days=1))] == [tid]


def test_due_now_returns_only_unfired_uncancelled_past_due_and_mark_fired_removes_it():
    s = _store()
    a = s.add(
        label="a",
        due=NOW - timedelta(seconds=1),
        asked_by="",
        obs_id=None,
        passes_quiet=True,
        now=NOW - timedelta(minutes=5),
    )
    b = s.add(
        label="b",
        due=NOW + timedelta(minutes=1),
        asked_by="",
        obs_id=None,
        passes_quiet=False,
        now=NOW,
    )
    c = s.add(
        label="c",
        due=NOW - timedelta(minutes=1),
        asked_by="",
        obs_id=None,
        passes_quiet=False,
        now=NOW - timedelta(minutes=5),
    )
    s.cancel(c, now=NOW)
    due = s.due_now(now=NOW)
    assert [r["id"] for r in due] == [a] and due[0]["passes_quiet"] is True
    s.mark_fired(a, now=NOW)
    assert s.due_now(now=NOW) == []
    assert [r["id"] for r in s.active(now=NOW)] == [b]


def test_cancel_all_and_recently_fired():
    s = _store()
    a = s.add(
        label="a",
        due=NOW + timedelta(minutes=1),
        asked_by="",
        obs_id=None,
        passes_quiet=False,
        now=NOW,
    )
    s.add(
        label="b",
        due=NOW + timedelta(minutes=2),
        asked_by="",
        obs_id=None,
        passes_quiet=False,
        now=NOW,
    )
    f = s.add(
        label="f",
        due=NOW - timedelta(minutes=1),
        asked_by="",
        obs_id=None,
        passes_quiet=False,
        now=NOW - timedelta(minutes=2),
    )
    s.mark_fired(f, now=NOW - timedelta(minutes=1))
    assert s.cancel_all(now=NOW) == 2
    assert s.active(now=NOW) == []
    assert [r["id"] for r in s.recently_fired(now=NOW, within_sec=180)] == [f]
    assert s.recently_fired(now=NOW + timedelta(minutes=10), within_sec=180) == []
    assert s.cancel(a, now=NOW) is False  # もう取り消してある
    assert s.cancel(999999, now=NOW) is False
