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
import json
import logging
import os
import threading
from datetime import date as _date, datetime
from typing import Any, Callable

import numpy as np

from ..mood_register import MoodPAD
from ..store import clock
from .embedding import _encode_vector

logger = logging.getLogger(__name__)

_CONTENT_DEDUP_WINDOW_SECS: int = int(os.environ.get("MEMORY_DEDUP_WINDOW_SECS", "30"))
_THUMB_SIZE = (320, 240)  # 画像は保存前にこの大きさへ縮小する（memory.py からの移動）


def _encode_image(image_path: str) -> str | None:
    try:
        import base64, io
        from PIL import Image
        with Image.open(image_path) as img:
            img.thumbnail(_THUMB_SIZE, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=60)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning("Failed to encode image %s: %s", image_path, e)
        return None


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


class ObservationWriteMixin:
    """観測の書き込みの持ち主。`ObservationMemory` が継承する。

    O は追記である（[D-O書込]）。`_materialize_save_event` はキューに積まれた
    イベントを実体化する本体で、埋め込みと situated 行の生成もここから起こす。

    宿主から借りる道具を下に宣言してある。これがこの層の依存の全てである。
    """

    # 宿主（ObservationMemory）が備えるもの。mixin 自身は持たない。
    _db_lock: threading.Lock
    _person_id: str
    _embedder: Any

    def _ensure_connected(self) -> Any: ...  # 宿主が実装する

    # 保存の一部として起こす、他の層の仕事。型の宣言だけにとどめる（メソッドとして
    # 定義すると MRO で本物より先に見つかり、実行時にこちらが呼ばれてしまう）。
    _refresh_situated_embeddings: Callable[..., None]
    _update_perspective_vec: Callable[..., None]
    _project_observation: Callable[..., None]

    def _materialize_save_event(
        self,
        event_id: str,
        payload: dict,
        writer_id: str | None = None,
        subject_id: str | None = None,
        participants: list[str] | None = None,
        scope: str = "speaker",
    ) -> bool:
        content   = str(payload.get("content", "")).strip()
        direction = str(payload.get("direction", "unknown"))
        kind      = str(payload.get("kind", "observation"))
        emotion   = str(payload.get("emotion", "neutral"))
        image_path = payload.get("image_path")
        override_date = payload.get("override_date")
        # PAD は payload 経由（to_json_dict/from_json_dict）。未指定は中立（列既定と同値）。
        # 呼び出し側の PAD 引き渡しは W2b-2。
        pad_dict = payload.get("emotion_pad")
        emotion_pad = MoodPAD.from_json_dict(pad_dict) if pad_dict else MoodPAD()

        if not content:
            return False

        image_data = _encode_image(image_path) if image_path else None
        vec = self._embedder.encode_document([content])[0]
        blob = _encode_vector(vec)
        # どの時計を使うかは store/clock.py に集約してある。
        now = clock.now_utc()
        save_ts = clock.end_of_day_utc(override_date) if override_date else now

        participants_json = json.dumps(participants or [], ensure_ascii=False)

        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM observations WHERE id=%s", (event_id,))
                if cur.fetchone():
                    return True
                if _CONTENT_DEDUP_WINDOW_SECS > 0:
                    cur.execute(
                        "SELECT id FROM observations "
                        "WHERE person_id = %s AND content = %s AND kind = %s "
                        "  AND timestamp >= now() - (%s * INTERVAL '1 second') "
                        "  AND superseded_by IS NULL "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (self._person_id, content, kind, _CONTENT_DEDUP_WINDOW_SECS),
                    )
                    if cur.fetchone():
                        logger.debug(
                            "content dedup skip: (person_id=%.8s kind=%s content=%.40r) "
                            "within %ds window",
                            self._person_id, kind, content, _CONTENT_DEDUP_WINDOW_SECS,
                        )
                        return True
                cur.execute(
                    "INSERT INTO observations "
                    "(id,content,timestamp,direction,kind,emotion,"
                    " image_path,image_data,person_id,writer_id,subject_id,"
                    " participants_json,scope,"
                    " emotion_p,emotion_pn,emotion_a,emotion_dom) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (event_id, content, save_ts,
                     direction, kind, emotion, image_path, image_data,
                     self._person_id,
                     writer_id or self._person_id,
                     subject_id or self._person_id,
                     participants_json, scope,
                     emotion_pad.p, emotion_pad.pn, emotion_pad.a, emotion_pad.dom),
                )
                cur.execute(
                    "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (event_id, blob),
                )
            # Pre-compute situated embeddings for all persons
            mem_vec = np.array(vec, dtype=np.float32)
            self._refresh_situated_embeddings(conn, event_id, mem_vec)
            self._project_observation(conn, event_id, content, kind, emotion)
            conn.commit()

        # Update this person's perspective vector in background
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                lambda: self._update_perspective_vec(self._person_id, np.array(vec, dtype=np.float32)),
            )
        except RuntimeError:
            self._update_perspective_vec(self._person_id, np.array(vec, dtype=np.float32))
        return True

    def _mark_recalled(self, ids: list[str], *, reinforce_half_life: bool) -> None:
        """Reinforce recalled memories by updating decay tracking columns."""
        if not ids:
            return
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    if reinforce_half_life:
                        cur.execute(
                            "UPDATE observations "
                            "SET recall_count = recall_count + 1, last_recalled_at = now() "
                            "WHERE id = ANY(%s)",
                            (ids,),
                        )
                    else:
                        cur.execute(
                            "UPDATE observations SET last_recalled_at = now() WHERE id = ANY(%s)",
                            (ids,),
                        )
                conn.commit()
        except Exception as e:
            logger.warning("_mark_recalled failed: %s", e)

    def decay_importance(self, before_date: str, factor: float = 0.95) -> int:
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE observations SET importance = importance * %s "
                    "WHERE timestamp::date < %s::date AND person_id = %s AND superseded_by IS NULL",
                    (factor, before_date, self._person_id),
                )
                count = cur.rowcount
            conn.commit()
        return count

    async def decay_importance_async(self, *a, **kw):
        return await asyncio.to_thread(self.decay_importance, *a, **kw)

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("UPDATE observations SET superseded_by=%s WHERE id=%s", (new_id, old_id))
            conn.commit()
