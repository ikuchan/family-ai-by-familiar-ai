"""アラームの器（知-q・2026-09-18・`設計方針_アラーム` v0.1）。表 `alarms`（067）。

タイマー（`store/timers.py`）とは別物。状態は列で持ち、content の時刻を読まない。再起動をまたいで残る。
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg2.extras

_COLS = "id, label, at, set_at, fired_at, cancelled_at, asked_by, obs_id, passes_quiet"


class AlarmStore:
    def __init__(self, conn) -> None:
        self._conn = conn

    def add(
        self,
        *,
        label: str,
        at: datetime,
        asked_by: str,
        obs_id: "str | None",
        passes_quiet: bool,
        now: "datetime | None" = None,
    ) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO alarms (label, at, set_at, asked_by, obs_id, passes_quiet) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (label, at, now, asked_by, obs_id, bool(passes_quiet)),
            )
            row = cur.fetchone()
        self._commit()
        return int(row["id"] if isinstance(row, dict) else row[0])

    def active(self, *, now: "datetime | None" = None) -> list[dict]:
        """掛かっているもの（未発火・未取消）。鳴る時刻の近い順。"""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM alarms WHERE fired_at IS NULL AND cancelled_at IS NULL "
                "ORDER BY at ASC, id ASC"
            )
            return [dict(r) for r in cur.fetchall()]

    def due_now(self, *, now: "datetime | None" = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM alarms WHERE at <= %s "
                "AND fired_at IS NULL AND cancelled_at IS NULL ORDER BY at ASC",
                (now,),
            )
            return [dict(r) for r in cur.fetchall()]

    def recently_fired(self, *, now: "datetime | None" = None, within_sec: float) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM alarms WHERE fired_at IS NOT NULL "
                "AND fired_at >= %s - make_interval(secs => %s) ORDER BY fired_at DESC",
                (now, float(within_sec)),
            )
            return [dict(r) for r in cur.fetchall()]

    def mark_fired(self, alarm_id: int, *, now: "datetime | None" = None) -> bool:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alarms SET fired_at = %s WHERE id = %s AND fired_at IS NULL AND cancelled_at IS NULL",
                (now, int(alarm_id)),
            )
            n = cur.rowcount
        self._commit()
        return bool(n)

    def cancel(self, alarm_id: int, *, now: "datetime | None" = None) -> bool:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alarms SET cancelled_at = %s WHERE id = %s AND fired_at IS NULL AND cancelled_at IS NULL",
                (now, int(alarm_id)),
            )
            n = cur.rowcount
        self._commit()
        return bool(n)

    def cancel_all(self, *, now: "datetime | None" = None) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alarms SET cancelled_at = %s WHERE fired_at IS NULL AND cancelled_at IS NULL",
                (now,),
            )
            n = cur.rowcount
        self._commit()
        return int(n or 0)

    def _cursor(self):
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _commit(self) -> None:
        if not getattr(self._conn, "autocommit", False):
            self._conn.commit()
