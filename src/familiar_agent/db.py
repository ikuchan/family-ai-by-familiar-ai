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


def apply_or_hold_migrations(conn, migration_dir=None) -> int:
    """マイグレーションを適用する。ただし `HOLD_MIGRATIONS` があれば留める。

    接続のたびに無条件で適用していたため、`migration/` へファイルを置いた時点で、次に
    その DB へ繋いだプロセスが適用してしまい、適用の可否を人が握れなかった（列改名の
    マイグレーションが意図せず本番へ入った）。起動時に `HOLD_MIGRATIONS=1` を渡せば
    留められる。スクリプトへは書かない（書くと留めた状態が既定になる）。

    **留めても起動は続く。** 未適用のまま動くと、コードが新しいスキーマを前提にして
    いる場合に実行時へ跳ねるので、何が保留されているかを1件ずつ warning に残す。件数
    だけでは、どれが保留かが分からない。保留が無いときは鳴らさない（毎回鳴ると読まれ
    なくなる）。

    **本体がマイグレーションを当てる口はここ 1 つ**で、呼ぶのは `Database.conn()` が新しく接続するとき（起動と、
    切れた接続の張り直し）だけ（環-av）。ほかの層から `apply_migrations` を直に呼ばない（記憶の層が接続を借りるたびに
    呼んでいたため、留めて起動しても置いたファイルがすぐ本番へ入った）。留めないときは、張り直しのときにも当たる。
    """
    from .db_migrations import (
        apply_migrations,
        default_migration_dir,
        pending_migration_ids,
    )

    mig_dir = migration_dir or default_migration_dir()
    if not os.getenv("HOLD_MIGRATIONS"):
        return apply_migrations(conn, mig_dir)

    pending = pending_migration_ids(conn, mig_dir)
    for mid in pending:
        logger.warning("マイグレーションを保留した（HOLD_MIGRATIONS）：%s", mid)
    if pending:
        logger.warning(
            "保留 %d 件。スキーマが古いままなので、新しい列を前提にした処理は落ちる",
            len(pending),
        )
    return 0


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
                    i + 1,
                    attempts,
                    delay,
                    e,
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


def under_db_lock(cls):
    """共有の接続を使う保存先の公開メソッドを、すべて共有の鍵の中で動かす（2026-11-23）。

    保存先（見えの普通・タイマー・アラームなど）は共有の接続を渡されて確定するが、鍵を取っていなかった。記憶の保存が
    別のスレッドで鍵を持って書いている途中に確定し、セーブポイントを消して記録を落とした（11/23 09:40）。呼び手ごとに
    鍵を取らせると取り忘れるので、保存先の側で 1 度に包む。
    """
    import functools

    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not callable(fn):
            continue

        def wrap(f):
            @functools.wraps(f)
            def locked(*a, **kw):
                with get_db().lock:
                    return f(*a, **kw)

            return locked

        setattr(cls, name, wrap(fn))
    return cls


class Database:
    """Thread-safe PostgreSQL connection wrapper (singleton)."""

    def __init__(self) -> None:
        self._conn: "psycopg2.extensions.connection | None" = None
        # 同じスレッドなら入れ子で取れる鍵（`under_db_lock` の保存先が、公開メソッドからほかの公開メソッドを呼んでも
        # 固まらない）。ほかのスレッドから見れば、いままでどおり 1 度に 1 つだけ。
        self._lock = threading.RLock()

    @property
    def lock(self) -> "threading.RLock":
        return self._lock

    def conn(self) -> "psycopg2.extensions.connection":
        """Return the live connection, (re)connecting if needed."""
        if self._conn is None or self._conn.closed:
            # 既定の接続先は置かない（環-ae・2026-10-08）。以前は無ければ本番に落ち、環境変数を空にしたテストが
            # 本番の設定値を読んだ。アプリは起動時に `.env`（`config.py` の `load_dotenv`）から受け取る。
            url = os.environ.get("DATABASE_URL", "")
            if not url:
                raise RuntimeError("DATABASE_URL が無いので DB につながない。.env を確かめて")
            self._conn = _connect_with_retry(url, attempts=3, delay=1.0)
            self._conn.autocommit = False
            # timestamptz を生活時間（ローカル）で読むため、セッション TimeZone を
            # ローカルオフセットへ固定する。timestamp::date・EXTRACT・psycopg2 の返す
            # datetime がローカルになる。挿入（now_utc）と TEXT 列比較は非影響。
            from .store.clock import local_utc_offset

            with self._conn.cursor() as _cur:
                _cur.execute(f"SET TIME ZONE INTERVAL '{local_utc_offset()}' HOUR TO MINUTE")
                # 絞り込み付きベクトル検索の取りこぼし対策（pgvector 0.8 の反復スキャン）。
                # HNSW 索引は vector 単体に張られ、person_id や現行かどうかの絞り込みは
                # 索引が近傍候補を集めた「後」に当たる。同じ観測を人数分の視点で持つため
                # 候補の大半が落ち、母集合が数千件あっても 0〜1 件しか残らないことがある
                # （実機で 0 件を観測）。反復スキャンは、絞り込みを通った行が必要数に達する
                # まで走査を続ける。上限は max_scan_tuples（既定 20000）が抑える。
                _cur.execute("SET hnsw.iterative_scan = relaxed_order")
            self._conn.commit()
            # 当てるのはここだけ（新しく接続したとき・環-av）。落ちたら接続を持たない——持ったままだと次の
            # `conn()` は当て直さず、古いスキーマのまま動き続ける（最初の接続は例外を DEBUG で飲む口から来うる）。
            try:
                apply_or_hold_migrations(self._conn)
            except Exception:
                self._conn.close()
                self._conn = None
                raise
            logger.debug("PostgreSQL connection established")
        elif self._conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            # Recover from a failed transaction so subsequent queries don't all fail.
            try:
                self._conn.rollback()
            except Exception as exc:
                # 失敗トランザクションからの回復。rollback 自体が失敗しても続行するが、
                # 無音にはしない（接続不調の兆候を残す・ログ方針）。
                logger.debug(
                    "rollback after failed transaction did not succeed: %s", exc, exc_info=True
                )
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
                    logger.debug(
                        "connection commit/close during teardown failed: %s", exc, exc_info=True
                    )
                finally:
                    self._conn = None


def vec_to_sql(vec: "list[float]") -> str:
    """Convert a float list to pgvector literal string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def sql_to_vec(s: str) -> "list[float]":
    """Parse pgvector string back to a float list."""
    return [float(x) for x in s.strip("[]").split(",")]
