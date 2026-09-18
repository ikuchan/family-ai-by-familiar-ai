"""ストップウォッチの器（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。表 `stopwatches`（068）。

タイマー（`store/timers.py`）・アラーム（`store/alarms.py`）とは別物。状態は列で持つ。再起動をまたいで残る。
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg2.extras

_COLS = "id, label, started_at, stopped_at, asked_by, obs_id, expired"


class StopwatchStore:
    def __init__(self, conn) -> None:
        self._conn = conn

    def add(
        self, *, label: str, asked_by: str, obs_id: "str | None", now: "datetime | None" = None
    ) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO stopwatches (label, started_at, asked_by, obs_id) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (label, now, asked_by, obs_id),
            )
            row = cur.fetchone()
        self._commit()
        return int(row["id"] if isinstance(row, dict) else row[0])

    def active(self, *, now: "datetime | None" = None) -> list[dict]:
        """動いているもの（未停止）。古い順。"""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM stopwatches WHERE stopped_at IS NULL ORDER BY started_at ASC, id ASC"
            )
            return [dict(r) for r in cur.fetchall()]

    def recently_stopped(self, *, now: "datetime | None" = None, within_sec: float) -> list[dict]:
        """直前に止めたもの（「何分だった？」に答えるため）。"""
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM stopwatches WHERE stopped_at IS NOT NULL "
                "AND stopped_at >= %s - make_interval(secs => %s) ORDER BY stopped_at DESC",
                (now, float(within_sec)),
            )
            return [dict(r) for r in cur.fetchall()]

    def stop(self, watch_id: int, *, now: "datetime | None" = None, expired: bool = False) -> bool:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE stopwatches SET stopped_at = %s, expired = %s "
                "WHERE id = %s AND stopped_at IS NULL",
                (now, bool(expired), int(watch_id)),
            )
            n = cur.rowcount
        self._commit()
        return bool(n)

    def stop_all(self, *, now: "datetime | None" = None) -> int:
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute("UPDATE stopwatches SET stopped_at = %s WHERE stopped_at IS NULL", (now,))
            n = cur.rowcount
        self._commit()
        return int(n or 0)

    def expire(self, *, now: "datetime | None" = None, max_sec: float) -> list[dict]:
        """寿命（`max_sec`）を超えて動いているものを止め、止めた行を返す（T が毎 tick）。"""
        now = now or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE stopwatches SET stopped_at = %s, expired = true "
                "WHERE stopped_at IS NULL AND started_at <= %s - make_interval(secs => %s) "
                f"RETURNING {_COLS}",
                (now, now, float(max_sec)),
            )
            rows = [dict(r) for r in cur.fetchall()]
        self._commit()
        return rows

    def _cursor(self):
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _commit(self) -> None:
        if not getattr(self._conn, "autocommit", False):
            self._conn.commit()
