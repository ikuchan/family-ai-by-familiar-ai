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
from datetime import date as _date, datetime
from typing import Any

import numpy as np

from ..db import vec_to_sql
from ..mood_register import MoodPAD
from ..person_memory_manager import AGENT_SELF_ID
from ..store import clock
from .context import StoreContext
from .embedding import _decode_vector, _encode_vector

logger = logging.getLogger(__name__)

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


class ObservationStore:
    """O（観測）の持ち主。読み出しと書き込みの両方を持つ。

    使うものは文脈（`StoreContext`）から受け取り、層をまたぐ依存は引数で受け取る。
    保存の一部として situated の更新と（撤去予定の）意味層への投影を起こすので、
    その2つを渡してもらう。
    """

    def __init__(self, ctx: StoreContext, *, situated: Any, legacy: Any) -> None:
        self._ctx = ctx
        self._situated = situated
        self._legacy = legacy


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
            with self._ctx.lock:
                conn = self._ctx.conn()
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
            with self._ctx.lock:
                conn = self._ctx.conn()
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

    def by_vector(
        self,
        query_vector_sql: str,
        n: int,
        *,
        kind: str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """situated 相関のコサイン順に n 件読む。**採点も足切りもしない**。

        返り行の `score` は**生のコサイン**で、5軸の合成スコアではない。合成と
        その足切りは W の構築（想起）の仕事で、層は持たない（[D-データモデル]）。
        呼び出し側は足切りを課すぶんだけ n を増やして取り、採点後に絞る。

        `query_vector_sql` は pgvector の文字列表現を受け取る。ベクトルの作り方
        （視点合成・平均中心化）は呼び出し側の責任で、層は受け取った表現で引くだけ。
        """
        kind_clause = "AND o.kind = %s" if kind else ""
        # 自分が出した検索が自分自身を拾わないようにする（id で狭く除外する）。
        exclude_clause = "AND NOT (o.id = ANY(%s))" if exclude_ids else ""
        params: list = [query_vector_sql, self._ctx.person_id]
        if kind:
            params.append(kind)
        if exclude_ids:
            params.append(list(exclude_ids))
        params += [query_vector_sql, n]
        # dumb 層は失敗を握り潰さない。例外は呼び出し側（recall）へ上げ、そこで方針
        # （loud に残す・degrade して []・keyword_fallback へ流さない）を持つ。0件と
        # 失敗を `[]` で混同すると、壊れた埋め込みパイプラインが keyword 検索で動いて
        # 見える masking になるため（棚卸し A1）。
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT o.id, o.content, o.timestamp,
                           o.direction, o.kind, o.emotion, o.image_path,
                           COALESCE(o.activation_a0, 1.0) AS activation_a0,
                           COALESCE(o.activation_n, 0) AS activation_n,
                           COALESCE(o.recall_count, 0) AS recall_count,
                           o.last_recalled_at,
                           o.emotion_p, o.emotion_pn, o.emotion_a, o.emotion_dom,
                           1 - (s.vector <=> %s::vector) AS score
                    FROM situated_embeddings s
                    JOIN observations o ON o.id = s.obs_id
                    WHERE s.person_id = %s
                      AND o.superseded_by IS NULL
                      {kind_clause}
                      {exclude_clause}
                    ORDER BY s.vector <=> %s::vector
                    LIMIT %s
                    """,
                    params,
                )
                return list(cur.fetchall())

    def by_recency(
        self,
        n: int,
        *,
        kind: str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """新しい順に n 件読む（新しさ軸の一次絞り）。**採点も足切りもしない**。

        設計 [D-想起合成] の**多軸 union 一次絞り**の一本。重み>0 の各軸で
        `ORDER BY … LIMIT N` を出して UNION し、和集合を再採点する、と定めてある。
        関連軸（`by_vector`）だけで候補を作っていたため、話題が近くない限り直近の
        記録が候補にすら入らず、t 軸は並べ替えにしか効いていなかった。

        返り行は `by_vector` と同じ列に揃える（`score` は持たない。関連は呼び出し側が
        `situated_cosines` で後から補う）。
        """
        kind_clause = "AND o.kind = %s" if kind else ""
        exclude_clause = "AND NOT (o.id = ANY(%s))" if exclude_ids else ""
        params: list = [self._ctx.person_id]
        if kind:
            params.append(kind)
        if exclude_ids:
            params.append(list(exclude_ids))
        params.append(n)
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT o.id, o.content, o.timestamp,
                           o.direction, o.kind, o.emotion, o.image_path,
                           COALESCE(o.activation_a0, 1.0) AS activation_a0,
                           COALESCE(o.activation_n, 0) AS activation_n,
                           COALESCE(o.recall_count, 0) AS recall_count,
                           o.last_recalled_at,
                           o.emotion_p, o.emotion_pn, o.emotion_a, o.emotion_dom
                    FROM situated_embeddings s
                    JOIN observations o ON o.id = s.obs_id
                    WHERE s.person_id = %s
                      AND o.superseded_by IS NULL
                      {kind_clause}
                      {exclude_clause}
                    ORDER BY o.timestamp DESC
                    LIMIT %s
                    """,
                    params,
                )
                return list(cur.fetchall())

    def content_novelty(self, mem_vec, conn, *, k: int, default: float) -> float:
        """内容の新規性 novelty ∈ [0,1]（課題5 v0.26）。

        **視点は常に AGENT_SELF**（a0/A はエージェント自身の活性・喚起・話者ではない）。
        内容を AGENT_SELF 視点で situate し、AGENT_SELF スコープの situated 近傍 K 件の
        コサイン平均の裏返し（1−平均）＝関連 r の鏡。**self_model（自己認識 MI）は母集合
        から除く**。近傍が K 未満なら既定（初期の横並びを避ける）。
        """
        try:
            v_sit = self._situated.situate(mem_vec, AGENT_SELF_ID, conn)
            q_sql = vec_to_sql(v_sit.tolist())
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 - (s.vector <=> %s::vector) AS c "
                    "FROM situated_embeddings s JOIN observations o ON o.id = s.obs_id "
                    "WHERE s.person_id = %s AND o.superseded_by IS NULL "
                    "  AND o.kind <> 'self_model' "
                    "ORDER BY s.vector <=> %s::vector LIMIT %s",
                    (q_sql, AGENT_SELF_ID, q_sql, k),
                )
                cosines = [float(row["c"]) for row in cur.fetchall()]
        except Exception as e:
            logger.warning("content_novelty failed, using default: %s", e)
            return default
        if len(cosines) < k:
            return default
        return max(0.0, min(1.0, 1.0 - sum(cosines) / len(cosines)))

    def situated_cosines(
        self, query_vector_sql: str, obs_ids: list[str], person_id: str,
    ) -> dict[str, float]:
        """指定 obs_id 群について、person_id 視点の situated コサインを返す（[D-在席相関]）。

        在席者相関 p の素点用。ベクトルの作り方（視点合成・平均中心化）は呼び出し側の
        責任で、層は受け取った表現で `situated_embeddings` を person_id 絞りで引くだけ。
        該当 situated 行が無い obs_id は結果に含めない（呼び出し側で 0 相当に畳む）。
        """
        if not obs_ids:
            return {}
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT s.obs_id, 1 - (s.vector <=> %s::vector) AS c "
                        "FROM situated_embeddings s JOIN observations o ON o.id = s.obs_id "
                        "WHERE s.person_id = %s AND o.superseded_by IS NULL "
                        "  AND s.obs_id = ANY(%s)",
                        (query_vector_sql, person_id, list(obs_ids)),
                    )
                    return {row["obs_id"]: float(row["c"]) for row in cur.fetchall()}
        except Exception as e:  # noqa: BLE001
            logger.warning("situated_cosines failed (person=%s): %s", person_id, e)
            return {}

    def keyword_fallback(self, query: str, n: int, kind: str | None) -> list[dict]:
        keywords = [w for w in query.split() if len(w) > 1][:4]
        if not keywords:
            return self.recency_fallback(n, kind)
        cond = " OR ".join(["o.content LIKE %s"] * len(keywords))
        params: list = [f"%{kw}%" for kw in keywords]
        kind_clause = "AND o.kind = %s" if kind else ""
        if kind:
            params.append(kind)
        params += [self._ctx.person_id, n]
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT o.id, o.content, o.timestamp,
                           o.direction, o.kind, o.emotion, o.image_path
                    FROM observations o
                    WHERE ({cond}) {kind_clause}
                      AND o.person_id = %s
                      AND o.superseded_by IS NULL
                    ORDER BY o.timestamp DESC LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [
            {
                "memory_id": r["id"], "timestamp": r["timestamp"],
                "summary": r["content"],
                "date": clock.ts_to_date(r["timestamp"]), "time": clock.ts_to_time(r["timestamp"]),
                "direction": r["direction"], "kind": r["kind"],
                "source_kind": r["kind"], "emotion": r["emotion"],
                "image_path": r["image_path"],
                "confidence": 0.45, "retrieval_method": "keyword",
            }
            for r in rows
        ]

    def recency_fallback(self, n: int, kind: str | None) -> list[dict]:
        kind_clause = "AND o.kind = %s" if kind else ""
        params: list = [self._ctx.person_id]
        if kind:
            params.append(kind)
        params.append(n)
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT o.id, o.content, o.timestamp, "
                        f"o.direction, o.kind, o.emotion, o.image_path "
                        f"FROM observations o "
                        f"WHERE o.person_id = %s AND o.superseded_by IS NULL {kind_clause} "
                        f"ORDER BY o.timestamp DESC LIMIT %s",
                        params,
                    )
                    rows = cur.fetchall()
            return [
                {
                    "memory_id": r["id"], "timestamp": r["timestamp"],
                    "summary": r["content"],
                    "date": clock.ts_to_date(r["timestamp"]), "time": clock.ts_to_time(r["timestamp"]),
                    "direction": r["direction"], "kind": r["kind"],
                    "source_kind": r["kind"], "emotion": r["emotion"],
                    "image_path": r["image_path"],
                    "confidence": 0.3, "retrieval_method": "recency",
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("recency_fallback failed: %s", e); return []

    def find_near_duplicates(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        """Return pairs of non-superseded observations whose vectors are >= threshold similar."""
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT e.obs_id, e.vector FROM obs_embeddings e "
                        "JOIN observations o ON o.id = e.obs_id "
                        "WHERE o.superseded_by IS NULL"
                    )
                    rows = cur.fetchall()
            if len(rows) < 2:
                return []
            ids = [r["obs_id"] for r in rows]
            vecs = np.array([_decode_vector(bytes(r["vector"])) for r in rows], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms > 1e-8, norms, 1.0)
            # 全ペア類似度は BLAS の行列積で一括計算し、上三角（i<j）だけ取る。
            # Python の O(n^2) 二重ループを避ける（出力は同一）。
            sims = vecs @ vecs.T
            iu, ju = np.triu_indices(len(ids), k=1)
            mask = sims[iu, ju] >= threshold
            return [
                (ids[int(i)], ids[int(j)], float(sims[int(i), int(j)]))
                for i, j in zip(iu[mask], ju[mask])
            ]
        except Exception:
            # 失敗（復号・OOM 等）はトレース付きで loud に残す。idle 統合を落とさず [] で degrade。
            logger.exception("find_near_duplicates failed")
            return []

    def pick_seed_candidates(
        self,
        hour: int,
        month: int,
        *,
        hour_window: int,
        month_window: int,
        k: int,
    ) -> list[dict]:
        """Return mixed seed candidates for associative memory sharing (Issue C).

        Three sub-pools are merged (deduped by id):
          - hour-near:   rows whose hour is within hour_window of `hour` (circular)
          - month-near:  rows whose month is within month_window of `month` (circular)
          - random:      any k rows
        Each sub-pool uses ORDER BY RANDOM() LIMIT k for lightweight diversity.
        time-of-day / seasonal proximity replaces the old time-label cosine query.
        """
        _COMMON = (
            "WHERE person_id=%s AND superseded_by IS NULL AND kind != 'day_summary' "
        )
        sql_hour = (
            "SELECT id, content, timestamp FROM observations " + _COMMON +
            "AND LEAST(ABS(EXTRACT(HOUR FROM timestamp)-%s), "
            "          24-ABS(EXTRACT(HOUR FROM timestamp)-%s)) <= %s "
            "ORDER BY RANDOM() LIMIT %s"
        )
        sql_month = (
            "SELECT id, content, timestamp FROM observations " + _COMMON +
            "AND LEAST(ABS(EXTRACT(MONTH FROM timestamp)-%s), "
            "          12-ABS(EXTRACT(MONTH FROM timestamp)-%s)) <= %s "
            "ORDER BY RANDOM() LIMIT %s"
        )
        sql_rand = (
            "SELECT id, content, timestamp FROM observations " + _COMMON +
            "ORDER BY RANDOM() LIMIT %s"
        )
        pid = self._ctx.person_id
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                seen: dict[str, dict] = {}
                with conn.cursor() as cur:
                    if hour_window > 0:
                        cur.execute(sql_hour, (pid, hour, hour, hour_window, k))
                        for r in cur.fetchall():
                            seen.setdefault(r["id"], dict(r))
                    if month_window > 0:
                        cur.execute(sql_month, (pid, month, month, month_window, k))
                        for r in cur.fetchall():
                            seen.setdefault(r["id"], dict(r))
                    cur.execute(sql_rand, (pid, k))
                    for r in cur.fetchall():
                        seen.setdefault(r["id"], dict(r))
            return list(seen.values())
        except Exception as e:
            logger.warning("pick_seed_candidates failed: %s", e)
            return []


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
            with self._ctx.lock:
                conn = self._ctx.conn()
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
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT timestamp::date AS d FROM observations "
                        "WHERE person_id=%s AND timestamp::date >= %s::date AND kind != 'day_summary' "
                        "ORDER BY d DESC",
                        (self._ctx.person_id, cutoff),
                    )
                    return [row["d"].isoformat() for row in cur.fetchall()]
        except Exception as e:
            logger.warning("get_dates_with_observations failed: %s", e); return []

    def get_dates_with_summaries(self) -> list[str]:
        """Return distinct dates (YYYY-MM-DD) that already have a day_summary observation."""
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT timestamp::date AS d FROM observations "
                        "WHERE person_id=%s AND kind='day_summary' ORDER BY d DESC",
                        (self._ctx.person_id,),
                    )
                    return [row["d"].isoformat() for row in cur.fetchall()]
        except Exception as e:
            logger.warning("get_dates_with_summaries failed: %s", e); return []

    def get_observations_for_date(self, date: str, limit: int = 50) -> list[dict]:
        """Return observations for a specific date (YYYY-MM-DD), oldest first."""
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, content, emotion, kind, timestamp "
                        "FROM observations "
                        "WHERE person_id=%s AND timestamp::date=%s::date AND kind != 'day_summary' "
                        "ORDER BY timestamp ASC LIMIT %s",
                        (self._ctx.person_id, date, limit),
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
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM observations WHERE kind='day_summary' AND timestamp::date=%s::date AND person_id=%s",
                        (date, self._ctx.person_id),
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
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, content, timestamp, emotion, kind FROM observations "
                        "WHERE EXTRACT(MONTH FROM timestamp) = %s "
                        "  AND EXTRACT(DAY FROM timestamp) = %s "
                        "  AND timestamp::date < %s "
                        "  AND person_id = %s "
                        "  AND superseded_by IS NULL "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (month, day, today, self._ctx.person_id, n),
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
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT MIN(timestamp::date) AS earliest FROM observations WHERE person_id = %s AND superseded_by IS NULL",
                        (self._ctx.person_id,)
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


    def materialize_save_event(
        self,
        event_id: str,
        payload: dict,
        *,
        dedup_window_secs: int = 30,
        writer_id: str | None = None,
        subject_id: str | None = None,
        participants: list[str] | None = None,
        scope: str = "speaker",
        novelty_k: int = 7,
        novelty_w_n: float = 1.5,
        novelty_default: float = 0.5,
        novelty_a0_cap: float = 1.5,
    ) -> str | None:
        """この内容を保持する行の id を返す（重複スキップなら既存行の id）。失敗は None。

        書かれていない id を返すと、それを宛先にした supersede が「どこも指さない」壊れた
        記録になるため、重複時は必ず既にある行の id を返す。
        """
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
            return None

        image_data = _encode_image(image_path) if image_path else None
        vec = self._ctx.embedder.encode_document([content])[0]
        blob = _encode_vector(vec)
        # どの時計を使うかは store/clock.py に集約してある。
        now = clock.now_utc()
        save_ts = clock.end_of_day_utc(override_date) if override_date else now

        participants_json = json.dumps(participants or [], ensure_ascii=False)

        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM observations WHERE id=%s", (event_id,))
                if cur.fetchone():
                    return event_id
                if dedup_window_secs > 0:
                    cur.execute(
                        "SELECT id FROM observations "
                        "WHERE person_id = %s AND content = %s AND kind = %s "
                        "  AND timestamp >= now() - (%s * INTERVAL '1 second') "
                        "  AND superseded_by IS NULL "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (self._ctx.person_id, content, kind, dedup_window_secs),
                    )
                    _dup = cur.fetchone()
                    if _dup:
                        logger.debug(
                            "content dedup skip: (person_id=%.8s kind=%s content=%.40r) "
                            "within %ds window → 既存 %.8s を返す",
                            self._ctx.person_id, kind, content, dedup_window_secs, _dup["id"],
                        )
                        return str(_dup["id"])
                # 取込 novelty（内容の新規性）→ a0。挿入前に測るので新観測は母集合に
                # 居ない（=自分は含まれない）。視点は AGENT_SELF・self_model 除外。
                novelty = self.content_novelty(
                    np.asarray(vec, dtype=np.float32), conn,
                    k=novelty_k, default=novelty_default,
                )
                activation_a0 = max(0.0, min(novelty_a0_cap, novelty_w_n * novelty))
                cur.execute(
                    "INSERT INTO observations "
                    "(id,content,timestamp,direction,kind,emotion,"
                    " image_path,image_data,person_id,writer_id,subject_id,"
                    " participants_json,scope,activation_a0,parent_id,"
                    " emotion_p,emotion_pn,emotion_a,emotion_dom) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (event_id, content, save_ts,
                     direction, kind, emotion, image_path, image_data,
                     self._ctx.person_id,
                     writer_id or self._ctx.person_id,
                     subject_id or self._ctx.person_id,
                     participants_json, scope, activation_a0,
                     payload.get("parent_id"),
                     emotion_pad.p, emotion_pad.pn, emotion_pad.a, emotion_pad.dom),
                )
                cur.execute(
                    "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (event_id, blob),
                )
            # Pre-compute situated embeddings for all persons
            mem_vec = np.array(vec, dtype=np.float32)
            self._situated.refresh_situated_embeddings(conn, event_id, mem_vec)
            self._legacy.project_observation(conn, event_id, content, kind, emotion)
            conn.commit()

        # Update this person's perspective vector in background
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                lambda: self._situated.update_perspective_vec(self._ctx.person_id, np.array(vec, dtype=np.float32)),
            )
        except RuntimeError:
            self._situated.update_perspective_vec(self._ctx.person_id, np.array(vec, dtype=np.float32))
        return event_id

    def _mark_recalled(self, ids: list[str], *, reinforce_half_life: bool) -> None:
        """Reinforce recalled memories by updating decay tracking columns."""
        if not ids:
            return
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
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
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE observations SET importance = importance * %s "
                    "WHERE timestamp::date < %s::date AND person_id = %s AND superseded_by IS NULL",
                    (factor, before_date, self._ctx.person_id),
                )
                count = cur.rowcount
            conn.commit()
        return count

    async def decay_importance_async(self, *a, **kw):
        return await asyncio.to_thread(self.decay_importance, *a, **kw)

    def close_with_children(self, parent_id: str, new_id: str) -> None:
        """親を閉じ、生きている子も同じ記録で閉じる（親子2階層・一段だけ・再帰なし）。

        調査は複数並行しうるので、親（求め）に子（調査）がぶら下がる。答えが出て親が決着
        したら、その求めのために投げた調査はもう追わない。孫は作らない設計なので、子の子を
        辿る必要はない。解決は先着が勝つ（`mark_superseded` と同じく上書きしない）。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE observations SET superseded_by=%s "
                    "WHERE (id=%s OR parent_id=%s) AND superseded_by IS NULL AND id<>%s",
                    (new_id, parent_id, parent_id, new_id),
                )
            conn.commit()

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                # 解決は先着が勝つ。既に解決済みの行を張り替えると「どの記録が解決したか」の
                # つながりが失われる（重複スキップで同じ id を持つ側が後から解決を試みる）。
                cur.execute(
                    "UPDATE observations SET superseded_by=%s "
                    "WHERE id=%s AND superseded_by IS NULL",
                    (new_id, old_id),
                )
            conn.commit()
