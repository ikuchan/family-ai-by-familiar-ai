"""PostgreSQL connection management.

Single shared connection with threading.Lock — same pattern as the
original sqlite3 implementation, so all asyncio.to_thread() callers
are safe without any additional changes.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

from .errors import FatalStartupError

if TYPE_CHECKING:
    import psycopg2.extensions

logger = logging.getLogger(__name__)


def _mask_db_url(url: str) -> str:
    """DB URL のパスワードを伏せて返す（ログ・メッセージ用）。"""
    try:
        p = urlparse(url)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        db = (p.path or "/").lstrip("/") or "?"
        return f"{host}{port}/{db}"
    except Exception:  # noqa: BLE001
        return "?"


def _connect_with_retry(url: str, attempts: int = 3, delay: float = 1.0):
    """psycopg2.connect を最大 attempts 回試す（既定＝初回＋2リトライ）。失敗は致命。"""
    last: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return psycopg2.connect(url)
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                logger.warning(
                    "PostgreSQL 接続に失敗（%d/%d・%.0fs 後に再試行）: %s",
                    i + 1, attempts, delay, e,
                )
                if delay > 0:
                    time.sleep(delay)
    raise FatalStartupError(
        f"PostgreSQL に接続できません（{_mask_db_url(url)}）。"
        f"DB が起動しているか・DATABASE_URL を確認してください。詳細: {last}"
    )

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
            self._conn = _connect_with_retry(url, attempts=3, delay=1.0)
            self._conn.autocommit = False
            # timestamptz を生活時間（ローカル）で読むため、セッション TimeZone を
            # ローカルオフセットへ固定する。timestamp::date・EXTRACT・psycopg2 の返す
            # datetime がローカルになる。挿入（now_utc）と TEXT 列比較は非影響。
            from .store.clock import local_utc_offset
            with self._conn.cursor() as _cur:
                _cur.execute(f"SET TIME ZONE INTERVAL '{local_utc_offset()}' HOUR TO MINUTE")
                # 絞り込み付きベクトル検索の取りこぼし対策（pgvector 0.8 の反復スキャン）。
                # HNSW 索引は vector 単体に張られ、person_id や superseded_by の絞り込みは
                # 索引が近傍候補を集めた「後」に当たる。同じ観測を人数分の視点で持つため
                # 候補の大半が落ち、母集合が数千件あっても 0〜1 件しか残らないことがある
                # （実機で 0 件を観測）。反復スキャンは、絞り込みを通った行が必要数に達する
                # まで走査を続ける。上限は max_scan_tuples（既定 20000）が抑える。
                _cur.execute("SET hnsw.iterative_scan = relaxed_order")
            self._conn.commit()
            from .db_migrations import apply_migrations, default_migration_dir
            apply_migrations(self._conn, default_migration_dir())
            logger.debug("PostgreSQL connection established")
        elif self._conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            # Recover from a failed transaction so subsequent queries don't all fail.
            try:
                self._conn.rollback()
            except Exception as exc:
                # 失敗トランザクションからの回復。rollback 自体が失敗しても続行するが、
                # 無音にはしない（接続不調の兆候を残す・ログ方針）。
                logger.debug("rollback after failed transaction did not succeed: %s", exc, exc_info=True)
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
                except Exception as exc:
                    # テアダウン。commit/close が落ちても _conn は下の finally で捨てるが、
                    # 無音にはしない（ログ方針）。
                    logger.debug("connection commit/close during teardown failed: %s", exc, exc_info=True)
                finally:
                    self._conn = None


def vec_to_sql(vec: "list[float]") -> str:
    """Convert a float list to pgvector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def sql_to_vec(s: str) -> "list[float]":
    """Parse pgvector string back to a float list."""
    return [float(x) for x in s.strip("[]").split(",")]
