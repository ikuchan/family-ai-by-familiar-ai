"""PostgreSQL connection management.

Single shared connection with threading.Lock — same pattern as the
original sqlite3 implementation, so all asyncio.to_thread() callers
are safe without any additional changes.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

import psycopg2
import psycopg2.extras

if TYPE_CHECKING:
    import psycopg2.extensions

logger = logging.getLogger(__name__)

_INSTANCE: "Database | None" = None
_INSTANCE_LOCK = threading.Lock()


def get_db() -> "Database":
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = Database()
    return _INSTANCE


class Database:
    """Thread-safe PostgreSQL connection wrapper (singleton)."""

    def __init__(self) -> None:
        self._conn: "psycopg2.extensions.connection | None" = None
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def conn(self) -> "psycopg2.extensions.connection":
        """Return the live connection, (re)connecting if needed."""
        if self._conn is None or self._conn.closed:
            url = os.environ.get(
                "DATABASE_URL",
                "postgresql://familiar:familiar@localhost:5432/familiar_ai",
            )
            self._conn = psycopg2.connect(url)
            self._conn.autocommit = False
            # timestamptz を生活時間（ローカル）で読むため、セッション TimeZone を
            # ローカルオフセットへ固定する。timestamp::date・EXTRACT・psycopg2 の返す
            # datetime がローカルになる。挿入（now_utc）と TEXT 列比較は非影響。
            from .store.clock import local_utc_offset
            with self._conn.cursor() as _cur:
                _cur.execute(f"SET TIME ZONE INTERVAL '{local_utc_offset()}' HOUR TO MINUTE")
            self._conn.commit()
            from .db_migrations import apply_migrations, default_migration_dir
            apply_migrations(self._conn, default_migration_dir())
            logger.debug("PostgreSQL connection established")
        elif self._conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            # Recover from a failed transaction so subsequent queries don't all fail.
            try:
                self._conn.rollback()
            except Exception:
                pass
        return self._conn

    def cursor(self) -> "psycopg2.extras.RealDictCursor":
        """Return a RealDictCursor (dict-like rows, same as sqlite3.Row)."""
        return self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def commit(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            if self._conn and not self._conn.closed:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None


def vec_to_sql(vec: "list[float]") -> str:
    """Convert a float list to pgvector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def sql_to_vec(s: str) -> "list[float]":
    """Parse pgvector string back to a float list."""
    return [float(x) for x in s.strip("[]").split(",")]
