"""O（観測）の読み出し層。

課題8 v0.6 で「層は生 SQL や WHERE を受けず、取り出しパターンごとの専用メソッドで
構造化値を受ける」と決めた。取り出しのパターンは3つある。

- **by_kind**      種別と person で新しい順に読む
- **by_situated**  situated 相関で紐づく観測を読む（所有者に依らない母集合）
- **by_date**      日付・周期で読む（日ごとの一覧、その日の観測、記念日）

**層は採点も想起判断も持たない**（[D-データモデル]）。想起スコアは層の外（W の構築）
で付ける。ここに `min_score` や5軸の採点を持ち込むと、ストアの実体が機構へ漏れる。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import date as _date, datetime
from typing import Any

from ..store import clock

logger = logging.getLogger(__name__)


class ObservationReadMixin:
    """観測の読み出しの持ち主。`ObservationMemory` が継承する。

    宿主から借りる道具を下に宣言してある。これがこの層の依存の全てである。
    """

    # 宿主（ObservationMemory）が備えるもの。mixin 自身は持たない。
    _db_lock: threading.Lock
    _person_id: str

    def _ensure_connected(self) -> Any: ...  # 宿主が実装する


    def _read_observations_by_kind(
        self, kind: str | tuple[str, ...], person_id: str, n: int, columns: tuple[str, ...]
    ) -> list[dict]:
        """observations を kind と person_id で絞り、新しい順に n 件読む dumb な読み出し。

        kind は単一値（str）または複数値（tuple[str, ...]）。複数値のときは kind IN (...)。
        採点・想起判断・trigger 判断は持たない（機械的な読み出しのみ）。
        設計ドキュメントで確定したストアアクセス層の最初の実体。
        行(dict)のリストを返す。失敗時は空リスト。
        """
        col_sql = ", ".join(columns)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    if isinstance(kind, str):
                        cur.execute(
                            f"SELECT {col_sql} FROM observations "
                            "WHERE kind=%s AND person_id=%s "
                            "ORDER BY timestamp DESC LIMIT %s",
                            (kind, person_id, n),
                        )
                    else:
                        placeholders = ", ".join(["%s"] * len(kind))
                        cur.execute(
                            f"SELECT {col_sql} FROM observations "
                            f"WHERE kind IN ({placeholders}) AND person_id=%s "
                            "ORDER BY timestamp DESC LIMIT %s",
                            (*kind, person_id, n),
                        )
                    return list(cur.fetchall())
        except Exception as e:
            logger.warning("_read_observations_by_kind failed: %s", e); return []

    def _read_observations_by_situated(
        self, person_id: str, n: int, columns: tuple[str, ...],
        *, kind: str | None = None, keywords: tuple[str, ...] = (),
    ) -> list[dict]:
        """observations を situated 相関で person に紐づけ、新しい順に n 件読む dumb な読み出し。

        所有者絞り（observations.person_id）でなく situated_embeddings を JOIN し
        s.person_id で紐づける（母集合はその person の視点で状況化された観測・所有者に依らない）。
        順序は timestamp DESC でベクトル類似度は使わない。kind と keywords（content LIKE の OR）は任意。
        採点・想起判断・trigger 判断は持たない。行(dict)のリストを返す。失敗時は空リスト。
        """
        col_sql = ", ".join(f"o.{c}" for c in columns)
        clauses = ["s.person_id=%s", "o.superseded_by IS NULL"]
        params: list = [person_id]
        if kind is not None:
            clauses.append("o.kind=%s")
            params.append(kind)
        if keywords:
            like_sql = " OR ".join(["o.content LIKE %s"] * len(keywords))
            clauses.append(f"({like_sql})")
            params += [f"%{kw}%" for kw in keywords]
        params.append(n)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {col_sql} FROM situated_embeddings s "
                        "JOIN observations o ON o.id = s.obs_id "
                        f"WHERE {' AND '.join(clauses)} "
                        "ORDER BY o.timestamp DESC LIMIT %s",
                        tuple(params),
                    )
                    return list(cur.fetchall())
        except Exception as e:
            logger.warning("_read_observations_by_situated failed: %s", e); return []

    def _read_supersede_chain(
        self, head_id: str, columns: tuple[str, ...]
    ) -> list[dict]:
        """現行版 MI（head_id）を起点に supersede の版チェーンを再構成する dumb な読み出し。

        `superseded_by`（旧→新を指す）を再帰でさかのぼり、head（depth 0）と祖先
        （旧版）を depth 昇順（新→旧）で返す。系統B 畳み込みの改訂履歴の再構成に使う
        （§7）。採点・想起判断は持たない。既存経路からは未接続。失敗時は空リスト。
        head_id が存在しなければ空リスト。
        """
        col_sql = ", ".join(f"o.{c}" for c in columns)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "WITH RECURSIVE chain AS ("
                        "  SELECT id, 0 AS depth FROM observations WHERE id=%s"
                        "  UNION ALL"
                        "  SELECT o.id, c.depth+1 FROM observations o "
                        "    JOIN chain c ON o.superseded_by = c.id"
                        ") "
                        f"SELECT {col_sql} FROM chain c JOIN observations o ON o.id = c.id "
                        "ORDER BY c.depth",
                        (head_id,),
                    )
                    return list(cur.fetchall())
        except Exception as e:
            logger.warning("_read_supersede_chain failed: %s", e); return []

    def get_dates_with_observations(self, days: int = 7) -> list[str]:
        """Return distinct dates (YYYY-MM-DD) that have observations within the last N days."""
        try:
            from datetime import timedelta
            cutoff = (datetime.fromisoformat(clock.now_local_iso()) - timedelta(days=days)).strftime("%Y-%m-%d")
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT timestamp::date AS d FROM observations "
                        "WHERE person_id=%s AND timestamp::date >= %s::date AND kind != 'day_summary' "
                        "ORDER BY d DESC",
                        (self._person_id, cutoff),
                    )
                    return [row["d"].isoformat() for row in cur.fetchall()]
        except Exception as e:
            logger.warning("get_dates_with_observations failed: %s", e); return []

    def get_dates_with_summaries(self) -> list[str]:
        """Return distinct dates (YYYY-MM-DD) that already have a day_summary observation."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT timestamp::date AS d FROM observations "
                        "WHERE person_id=%s AND kind='day_summary' ORDER BY d DESC",
                        (self._person_id,),
                    )
                    return [row["d"].isoformat() for row in cur.fetchall()]
        except Exception as e:
            logger.warning("get_dates_with_summaries failed: %s", e); return []

    def get_observations_for_date(self, date: str, limit: int = 50) -> list[dict]:
        """Return observations for a specific date (YYYY-MM-DD), oldest first."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, content, emotion, kind, timestamp "
                        "FROM observations "
                        "WHERE person_id=%s AND timestamp::date=%s::date AND kind != 'day_summary' "
                        "ORDER BY timestamp ASC LIMIT %s",
                        (self._person_id, date, limit),
                    )
                    rows = cur.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "content": row["content"],
                    "emotion": row["emotion"] or "neutral",
                    "kind": row["kind"] or "conversation",
                    "time": clock.ts_to_time(row["timestamp"]),
                })
            return result
        except Exception as e:
            logger.warning("get_observations_for_date failed: %s", e); return []

    def delete_day_summaries_for_date(self, date: str) -> int:
        """Delete all day_summary observations for a given date. Returns deleted row count."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM observations WHERE kind='day_summary' AND timestamp::date=%s::date AND person_id=%s",
                        (date, self._person_id),
                    )
                    count = cur.rowcount if hasattr(cur, "rowcount") else 0
                conn.commit()
            return count
        except Exception as e:
            logger.warning("delete_day_summaries_for_date failed: %s", e); return 0

    # -- Importance decay, supersession, links, episodes --
    # These methods follow the same person_id pattern.
    # Abbreviated here; full implementations mirror recall() with
    # AND person_id = %s added to every WHERE clause.

    def recall_on_this_day(self, month: int, day: int, n: int = 5) -> list[dict]:
        """Return observations from past years on the same month/day (anniversary recall)."""
        try:
            today = _date.today()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, content, timestamp, emotion, kind FROM observations "
                        "WHERE EXTRACT(MONTH FROM timestamp) = %s "
                        "  AND EXTRACT(DAY FROM timestamp) = %s "
                        "  AND timestamp::date < %s "
                        "  AND person_id = %s "
                        "  AND superseded_by IS NULL "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (month, day, today, self._person_id, n),
                    )
                    return [
                        {**dict(r), "date": clock.ts_to_date(r["timestamp"]), "time": clock.ts_to_time(r["timestamp"])}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_on_this_day failed: %s", e); return []

    async def recall_on_this_day_async(self, month: int, day: int, n: int = 5) -> list[dict]:
        return await asyncio.to_thread(self.recall_on_this_day, month, day, n)

    def get_earliest_date(self) -> str | None:
        """Return the earliest observation date string (YYYY-MM-DD), or None if no records."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT MIN(timestamp::date) AS earliest FROM observations WHERE person_id = %s AND superseded_by IS NULL",
                        (self._person_id,)
                    )
                    row = cur.fetchone()
                if row is None:
                    return None
                val = row["earliest"]
                return str(val) if val is not None else None
        except Exception as e:
            logger.warning("get_earliest_date failed: %s", e); return None

    async def get_earliest_date_async(self) -> str | None:
        return await asyncio.to_thread(self.get_earliest_date)
