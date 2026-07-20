"""Tests for store/clock.py（時刻の一元化）.

2026-07-20 に、`observations.timestamp`（timestamptz）へ tz を持たない
`datetime.now()` を入れて9時間ずれる不具合が出た。同じ表の `last_recalled_at` は
SQL の `now()` で書かれており、2つの時刻列が別の時計を指していた。

一方で `memory_events.created_at` や `memory_jobs.available_at` は TEXT 列で、
ローカル時刻の ISO 文字列どうしを比較している（`available_at <= now`）。ここを
UTC へ換算すると既存行との比較が壊れる。

つまり時計は用途で分かれる。**分かれていること自体を1箇所に書き出す**のが
このモジュールの役目で、換算はしない。
"""

from __future__ import annotations

from datetime import datetime, timezone

from familiar_agent.store.clock import end_of_day_utc, local_tz, now_local_iso, now_utc


def test_now_utc_is_timezone_aware() -> None:
    """timestamptz 列へ入れる時刻は tz を持つ（持たないと DB が UTC と誤解する）。"""
    got = now_utc()
    assert got.tzinfo is not None
    assert got.utcoffset() == timezone.utc.utcoffset(got)


def test_now_utc_is_close_to_real_time() -> None:
    got = now_utc()
    assert abs((got - datetime.now(timezone.utc)).total_seconds()) < 5


def test_now_local_iso_is_naive_local_text() -> None:
    """TEXT 列向けはローカル時刻の素の ISO。既存行との比較を保つため換算しない。"""
    got = now_local_iso()
    parsed = datetime.fromisoformat(got)
    assert parsed.tzinfo is None, "TEXT 列向けに tz 付きを返すと既存行と比較できない"
    assert abs((parsed - datetime.now()).total_seconds()) < 5


def test_local_iso_ordering_matches_string_ordering() -> None:
    """文字列比較で時系列になる（`available_at <= now` が成り立つ前提）。"""
    a = now_local_iso()
    b = now_local_iso()
    assert a <= b


def test_end_of_day_is_local_midnight_in_utc() -> None:
    """日付指定はその日の終わりを意味する。ローカルの 23:59:59 を UTC で返す。"""
    got = end_of_day_utc("2026-07-01")
    assert got.tzinfo is not None
    back = got.astimezone(local_tz())
    assert (back.year, back.month, back.day) == (2026, 7, 1)
    assert (back.hour, back.minute, back.second) == (23, 59, 59)
