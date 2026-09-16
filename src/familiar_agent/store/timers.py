"""タイマーの器（知-n・2026-09-15・`設計方針_タイマー` v0.1・065）。

アラーム・タイマー・ストップウォッチの状態を表 `timers` の**列で**持つ（content の時刻を読まない）。
`due` が NULL ならストップウォッチ。読む側は T（毎 tick `due_now`）と W の枠（`active`・`recently_fired`）、
書く側は道具（`add`・`cancel`）と T（`mark_fired`）。接続は呼び手が渡す（`db.lock` の中で使う）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import psycopg2.extras

_COLS = "id, label, due, started_at, fired_at, cancelled_at, asked_by, obs_id, passes_quiet"


class TimerStore:
    def __init__(self, conn) -> None:
        self._conn = conn

    def add(
        self,
        *,
        label: str,
        due: "datetime | None",
        asked_by: str,
        obs_id: "str | None",
        passes_quiet: bool,
        now: "datetime | None" = None,
    ) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO timers (label, due, started_at, asked_by, obs_id, passes_quiet) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (label, due, now, asked_by, obs_id, bool(passes_quiet)),
            )
            row = cur.fetchone()
        self._commit()
        return int(row["id"] if isinstance(row, dict) else row[0])

    def active(self, *, now: "datetime | None" = None) -> list[dict]:
        """動いているもの（未発火・未取消）。due の近い順、ストップウォッチは後ろ。"""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM timers WHERE fired_at IS NULL AND cancelled_at IS NULL "
                "ORDER BY due ASC NULLS LAST, id ASC"
            )
            return [dict(r) for r in cur.fetchall()]

    def due_now(self, *, now: "datetime | None" = None) -> list[dict]:
        """鳴らす頃合い（due を過ぎた未発火・未取消）。ストップウォッチは含まない。"""
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM timers WHERE due IS NOT NULL AND due <= %s "
                "AND fired_at IS NULL AND cancelled_at IS NULL ORDER BY due ASC",
                (now,),
            )
            return [dict(r) for r in cur.fetchall()]

    def recently_fired(self, *, now: "datetime | None" = None, within_sec: float) -> list[dict]:
        """直前に鳴ったもの（「止めて」に「もう止まっている」と答えるため）。"""
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM timers WHERE fired_at IS NOT NULL AND fired_at >= %s "
                "ORDER BY fired_at DESC",
                (now - timedelta(seconds=float(within_sec)),),
            )
            return [dict(r) for r in cur.fetchall()]

    def recently_stopped(self, *, now: "datetime | None" = None, within_sec: float) -> list[dict]:
        """直前に止めたもの（「何秒だった？」に答えるため・ストップウォッチの経過は止めた瞬間で決まる）。"""
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM timers WHERE cancelled_at IS NOT NULL AND cancelled_at >= %s "
                "ORDER BY cancelled_at DESC",
                (now - timedelta(seconds=float(within_sec)),),
            )
            return [dict(r) for r in cur.fetchall()]

    def mark_fired(self, timer_id: int, *, now: "datetime | None" = None) -> bool:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE timers SET fired_at = %s WHERE id = %s AND fired_at IS NULL AND cancelled_at IS NULL",
                (now, int(timer_id)),
            )
            n = cur.rowcount
        self._commit()
        return bool(n)

    def cancel(self, timer_id: int, *, now: "datetime | None" = None) -> bool:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE timers SET cancelled_at = %s WHERE id = %s AND fired_at IS NULL AND cancelled_at IS NULL",
                (now, int(timer_id)),
            )
            n = cur.rowcount
        self._commit()
        return bool(n)

    def cancel_all(self, *, now: "datetime | None" = None) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE timers SET cancelled_at = %s WHERE fired_at IS NULL AND cancelled_at IS NULL",
                (now,),
            )
            n = cur.rowcount
        self._commit()
        return int(n or 0)

    def _cursor(self):
        """行を dict で返す cursor（共有接続の既定は tuple・実機で `dict(r)` が落ちた・2026-09-15）。"""
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _commit(self) -> None:
        if not getattr(self._conn, "autocommit", False):
            self._conn.commit()
