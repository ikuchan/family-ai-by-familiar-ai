"""Observation and emotional memory for the embodied agent.

Built-in tools:
- remember(content, emotion, scope): store an observation in PostgreSQL.
  scope: "speaker" | "witnessed" | "scene" | "all".
- recall(hint, n): retrieve semantically similar memories via pgvector cosine similarity.
Storage: PostgreSQL + pgvector (situated_embeddings, bge-m3).
Memory is scoped per person via PersonMemoryManager (person_id).
Config: DATABASE_URL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date as _date, datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import MemoryConfig
from ..db import Database, get_db, vec_to_sql
from ..mood_register import MoodPAD, load_current_mood
from ..db_migrations import apply_migrations, default_migration_dir
from ..legacy.semantic_layer import LegacySemanticLayerMixin
from ..store import clock
from ..store.jobs import MemoryJobsMixin
from ..store.observations import ObservationReadMixin, ObservationWriteMixin
from ..store.situated import (  # noqa: F401  既存の呼び出し側が memory 経由で引くための再輸出
    SituatedVectorsMixin,
    _situated_vector,
    load_embedding_mean,
)
from ..store.db_compat import _RealDictConnWrapper, _SQLiteConnWrapper
from ..store.embedding import (  # noqa: F401  既存の呼び出し側が memory 経由で引くための再輸出
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    _coerce_to_embedding_dim,
    _cosine_similarity,
    _decode_vector,
    _EmbeddingModel,
    _encode_vector,
    _normalise,
)
from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID
from ..time_decay import DecayState

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager

logger = logging.getLogger(__name__)


DB_PATH_UNUSED = ""          # kept for API compatibility, ignored

# Time-window dedup: identical (person_id, content, kind) within this many seconds
# is treated as a duplicate and silently skipped. Set to 0 to disable.


def _to_epoch(dt) -> float | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _stretch_relevance(cosine: float, *, c_lo: float, c_hi: float) -> float:
    """r 軸＝コサインの固定係数 min-max 伸長（課題5 v0.24 D 節）。

    r = clip((cos - c_lo) / (c_hi - c_lo), 0, 1)。seed や候補数に依存しない。
    確定値は c_lo=0.0 / c_hi=1.0 で、このとき伸長は恒等になる（平均中心化後の
    実測でコサインが [0,1] にほぼ収まったため・根拠台帳 v0.7 §3）。恒等でも式を
    通すのは、両係数が Config で可変であり、値を変えたときに r の経路が一本で
    あることを保つためである。

    c_hi <= c_lo の縮退時は段階を作れないので、0除算を避けてステップ関数
    （cos >= c_hi で 1、そうでなければ 0）へ退化させる。
    """
    span = c_hi - c_lo
    if span <= 0.0:
        return 1.0 if cosine >= c_hi else 0.0
    return max(0.0, min(1.0, (cosine - c_lo) / span))


@dataclass(frozen=True)
class _ScoreParts:
    """想起スコアと各軸の内訳。`e` は項を外したとき None。"""

    score: float
    r: float
    t: float
    a: float
    m: float
    e: float | None


def _score_breakdown(
    cosine: float,
    ts,
    last_recalled_at,
    recall_count: int,
    activation_a0: float,
    activation_n: int,
    *,
    obs_pad: tuple[float, float, float, float] | None = None,
    mood_pad: tuple[float, float, float, float] | None = None,
    half_life_days: float,
    floor: float,
    c_lo: float = 0.0,
    c_hi: float = 1.0,
    w_r: float = 1.0,
    w_t: float = 1.0,
    w_e: float = 1.0,
    w_a: float = 1.5,
    sigma: float = 1.0,
) -> _ScoreParts:
    """想起スコアと各軸の内訳を返す（合成式の正本）。

    `_compute_final_score` はここから `score` を取り出すだけの薄い包みである。
    内訳を別に組み立てると式が二重になり、片方だけ直したときにずれるので、
    計算はこの関数だけが持つ。`e` は項を外したとき None になる。

    式と各軸の意味は `_compute_final_score` の docstring にある。
    """
    base = last_recalled_at if last_recalled_at is not None else ts
    origin = _to_epoch(base)
    if origin is None:
        origin = datetime.now(timezone.utc).timestamp()
    state = DecayState(
        origin_epoch=origin,
        half_life_seconds=half_life_days * 86400.0,
        floor=floor,
        reinforce_count=max(0, recall_count),
    )
    now_epoch = datetime.now(timezone.utc).timestamp()

    r = _stretch_relevance(cosine, c_lo=c_lo, c_hi=c_hi)
    t = state.score(now_epoch)
    a = _derive_activation(float(activation_a0), int(activation_n))

    e: float | None = None
    numerator = w_t * t + w_a * a
    denominator = w_t + w_a
    if obs_pad is not None and mood_pad is not None:
        e = _emotion_match(obs_pad, mood_pad, sigma=sigma)
        numerator += w_e * e
        denominator += w_e

    m = 1.0 if denominator <= 0.0 else numerator / denominator
    return _ScoreParts(score=(r ** w_r) * m, r=r, t=t, a=a, m=m, e=e)


def _compute_final_score(
    cosine: float,
    ts,
    last_recalled_at,
    recall_count: int,
    activation_a0: float,
    activation_n: int,
    *,
    obs_pad: tuple[float, float, float, float] | None = None,
    mood_pad: tuple[float, float, float, float] | None = None,
    half_life_days: float,
    floor: float,
    c_lo: float = 0.0,
    c_hi: float = 1.0,
    w_r: float = 1.0,
    w_t: float = 1.0,
    w_e: float = 1.0,
    w_a: float = 1.5,
    sigma: float = 1.0,
) -> float:
    """想起スコアのハイブリッド合成（課題5 v0.24 D 節・Phase 2 スライス3）。

        score = r^{w_r} × M,  M = (w_t·t + w_e·e + w_a·a) / (w_t + w_e + w_a)

    関連 r だけが乗算ゲート（段階的関連係数）で、t・e・a は加重平均で補償的に
    束ねる。純積ではないので、一軸が低いだけで記憶が消えることはない。加算部の
    重みが全0なら M=1（score は r だけ）。

    - r：`_stretch_relevance` でコサインを伸長した値。
    - t：`DecayState`（強化A＝recall_count で実効半減期が倍、強化B＝
      last_recalled_at を origin にして若返り）。時間減衰はこの軸に一元化し、
      importance の日次減衰は使わない（P-1・[D-想起合成]）。
    - e：`_emotion_match(obs_pad, mood_pad)`＝**今の気分と観測 PAD の距離**。
      記憶どうしの感情距離ではない（`感情ループ全体像` の `M → RECALL`）。
      mood_pad が None（mood を読めなかった経路）のときは e 項を分子分母から
      外す。中立0.5で埋めると「気分に一致する記憶」を偽って作ってしまうため。
      obs_pad が None のときも同様に外す。
    - a：`_derive_activation(a0, n)`（イベント駆動・時間では減らさない）。
    - p（在席者相関）は知覚待ちのため項ごと持たない。課題5 の「在席者ゼロなら
      w_p 項を外す」に一致する。

    係数は MemoryConfig から注入する。計算の実体は `_score_breakdown` にある
    （内訳ログと式を共有するため）。
    """
    return _score_breakdown(
        cosine, ts, last_recalled_at, recall_count, activation_a0, activation_n,
        obs_pad=obs_pad, mood_pad=mood_pad,
        half_life_days=half_life_days, floor=floor,
        c_lo=c_lo, c_hi=c_hi,
        w_r=w_r, w_t=w_t, w_e=w_e, w_a=w_a, sigma=sigma,
    ).score


# ── Helpers ────────────────────────────────────────────────────────────────

# 時刻の整形は store/clock.py が持つ。既存の呼び出し名はそのまま使えるようにする。
_ts_to_date = clock.ts_to_date
_ts_to_time = clock.ts_to_time






@dataclass
class PrimitiveMentalItem:
    emotion: object | None = None   # PAD または未設定。A-1では未設定(None)
    drive: object | None = None     # 5欠乏 または未設定。A-1では未設定(None)


@dataclass
class MentalItem(PrimitiveMentalItem):
    id: str = ""
    content: str = ""
    vector: object | None = None
    supersedes: str | None = None
    activation: float | None = None


def _row_to_mental_item(row) -> MentalItem:
    """観測行から MentalItem を組み立てる。

    Y（W2a）：行の PAD 列（emotion_p/pn/a/dom）を `MoodPAD` として emotion に載せる。
    列を SELECT していない呼び出しでも `row.get` 既定0.5で中立になり安全。これで
    評価器の PAD・行の列・MI 器の emotion が同じ `MoodPAD` で一本化する（B-3 の
    tif.py が emotion に MoodPAD を使うのと型が揃う）。drive・vector は後続で未設定。
    """
    return MentalItem(
        id=row["id"],
        content=row["content"],
        supersedes=row["superseded_by"],
        activation=row["importance"],
        emotion=MoodPAD(
            p=row.get("emotion_p", 0.5),
            pn=row.get("emotion_pn", 0.5),
            a=row.get("emotion_a", 0.5),
            dom=row.get("emotion_dom", 0.5),
        ),
        drive=None,
        vector=None,
    )


def _derive_activation(
    a0: float, n: int, *, floor: float = 0.0, c: float = 2.0,
    epsilon: float = 0.001, step: float = 0.33,
) -> float:
    """初期値 a0 と正味デルタ回数 n から活性 a を導出する。

    a0 を [floor, c] で正規化し、ε で両端に寄せてロジットで無限区間へ写し、
    n·step を足してロジスティックで [floor, c] へ戻す。n=0 なら a=a0。
    floor・c・ε・step は設定値（値の確定は課題5・Config から差し替え可）。
    """
    span = c - floor
    x0 = (a0 - floor) / span
    x0 = min(max(x0, epsilon), 1.0 - epsilon)
    u = math.log(x0 / (1.0 - x0)) + n * step
    y = 1.0 / (1.0 + math.exp(-u))
    return floor + span * y


def _emotion_match(
    obs_pad: tuple[float, float, float, float],
    mood_pad: tuple[float, float, float, float],
    *,
    sigma: float = 1.0,
    lambdas: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    epsilon: float = 0.001,
) -> float:
    """感情一致 e を導出する（課題5 v0.24・ガウシアン）。

    各 PAD 軸を ε で両端へ寄せロジットで元空間へ戻し、軸重み λ_i つきの
    二乗距離 D²=Σ λ_i (logit(x_obs)-logit(x_mood))² を作り、
    e=exp(-D²/(2σ²)) を返す。完全一致で e=1、遠いほど 0 へ。
    σ は Config（`recall_emotion_sigma`）から、λ_i・ε は既定値を使う。

    `mood_pad` には**今の気分**を渡す（記憶どうしの感情距離ではない）。
    `_compute_final_score` の加算部の一項として使う（スライス3 で接続）。
    """
    def _logit(x: float) -> float:
        x = min(max(x, epsilon), 1.0 - epsilon)
        return math.log(x / (1.0 - x))

    d2 = 0.0
    for xo, xm, lam in zip(obs_pad, mood_pad, lambdas):
        delta = _logit(xo) - _logit(xm)
        d2 += lam * delta * delta
    return math.exp(-d2 / (2.0 * sigma * sigma))


# ── ObservationMemory ──────────────────────────────────────────────────────

class ObservationMemory(
    LegacySemanticLayerMixin,
    MemoryJobsMixin,
    SituatedVectorsMixin,
    ObservationReadMixin,
    ObservationWriteMixin,
):
    """PostgreSQL-backed memory store scoped to one person_id."""

    _embedder: "_EmbeddingModel" = _EmbeddingModel()  # class-level default; tests can patch via __class__

    def __init__(
        self,
        db_path: str = DB_PATH_UNUSED,          # ignored, kept for compat
        model_name: str = EMBEDDING_MODEL,
        person_id: str = DEFAULT_PERSON_ID,
    ) -> None:
        self._person_id = person_id
        self._db: Database = get_db()
        self._db_lock = self._db.lock
        self._embedder = _EmbeddingModel(model_name)
        import os
        if os.environ.get("FAMILIAR_EMBEDDING_PREWARM", "1").lower() not in {"0","false","no","off"}:
            self._embedder.pre_warm()

    # ── Internals ──────────────────────────────────────────────────────────

    def is_embedding_ready(self) -> bool:
        return self._embedder.is_ready()

    def close(self) -> None:
        self._db.close()

    def _ensure_connected(self):
        if callable(getattr(self._db, "conn", None)):
            conn = self._db.conn()
            apply_migrations(conn, default_migration_dir())
            return _RealDictConnWrapper(conn)
        # Raw sqlite3 connection (used in tests via _memory_with_rows)
        return _SQLiteConnWrapper(self._db)

    @staticmethod
    def _now() -> str:
        """TEXT 列向けの現在時刻（store/clock.py の使い分けに従う）。"""
        return clock.now_local_iso()

    # ── Person management ──────────────────────────────────────────────────

    def register_person(self, name: str, display_name: str = "", person_id: str | None = None) -> str:
        pid = person_id or str(uuid.uuid4())
        now = self._now()
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM persons WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    return str(row["id"])
                cur.execute(
                    "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (pid, name, display_name or name, now, now),
                )
            conn.commit()
        return pid

    def list_persons(self) -> list[dict]:
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, display_name, created_at FROM persons ORDER BY created_at")
                return [dict(r) for r in cur.fetchall()]

    def for_person(self, person_id: str) -> "ObservationMemory":
        """Return a lightweight view of this memory scoped to another person."""
        obj = object.__new__(ObservationMemory)
        obj._person_id = person_id
        obj._db        = self._db
        obj._db_lock   = self._db_lock
        obj._embedder  = self._embedder
        return obj

    # ── Perspective vector ─────────────────────────────────────────────────

    # ── Event / job queue ──────────────────────────────────────────────────


    def save(
        self,
        content: str,
        direction: str = "unknown",
        kind: str = "observation",
        emotion: str = "neutral",
        image_path: str | None = None,
        override_date: str | None = None,
        dedupe_key: str | None = None,
        materialize_now: bool = True,
        writer_id: str | None = None,
        subject_id: str | None = None,
        participants: list[str] | None = None,
        scope: str = "speaker",
        emotion_pad: MoodPAD | None = None,
    ) -> bool:
        # PAD は payload へ dict で載せる（JSON 往復可・遅延マテリアライズも通る）。
        # 呼び出し側の PAD 引き渡しは W2b-2。未指定は中立で外部挙動不変。
        payload = dict(content=content, direction=direction, kind=kind,
                       emotion=emotion, image_path=image_path,
                       override_date=override_date,
                       emotion_pad=emotion_pad.to_json_dict() if emotion_pad else None)
        try:
            event_id: str | None = None
            try:
                event_id, created_new = self.append_memory_event(
                    "memory.save", payload, dedupe_key=dedupe_key,
                    queue_job=True, job_type="materialize_observation",
                )
                if dedupe_key and event_id and not created_new:
                    return True
                if not materialize_now and event_id:
                    return True
            except Exception as e:
                logger.warning("append_memory_event failed, continuing with direct save: %s", e)
            obs_id = event_id or str(uuid.uuid4())
            return self._materialize_save_event(
                obs_id, payload,
                writer_id=writer_id,
                subject_id=subject_id,
                participants=participants,
                scope=scope,
            )
        except Exception as e:
            logger.warning("save failed: %s", e)
            return False

    def save_with_id(self, content: str, **kwargs) -> tuple[str | None, bool]:
        _pad = kwargs.get("emotion_pad")
        payload = dict(content=content, direction=kwargs.get("direction","unknown"),
                       kind=kwargs.get("kind","observation"), emotion=kwargs.get("emotion","neutral"),
                       image_path=kwargs.get("image_path"), override_date=kwargs.get("override_date"),
                       emotion_pad=_pad.to_json_dict() if _pad else None)
        try:
            event_id, created_new = self.append_memory_event(
                "memory.save", payload,
                dedupe_key=kwargs.get("dedupe_key"),
                queue_job=True, job_type="materialize_observation",
            )
            if kwargs.get("dedupe_key") and event_id and not created_new:
                return event_id, True
            if not kwargs.get("materialize_now", True) and event_id:
                return event_id, True
            obs_id = event_id or str(uuid.uuid4())
            ok = self._materialize_save_event(
                obs_id, payload,
                writer_id=kwargs.get("writer_id"),
                subject_id=kwargs.get("subject_id"),
                participants=kwargs.get("participants"),
                scope=kwargs.get("scope", "speaker"),
            )
            return (obs_id if ok else None), ok
        except Exception as e:
            logger.warning("save_with_id failed: %s", e)
            return None, False

    async def save_async(self, *a, **kw) -> bool:
        return await asyncio.to_thread(self.save, *a, **kw)

    async def save_async_with_id(self, *a, **kw) -> tuple[str | None, bool]:
        return await asyncio.to_thread(self.save_with_id, *a, **kw)

    def recall(self, query: str, n: int = 3, kind: str | None = None,
               min_score: float = 0.0,
               recall_mode: str = "system") -> list[dict]:
        """Recall using situated vectors (pgvector cosine search).

        min_score:   cosine similarity threshold (Issue A).
        recall_mode: reinforcement mode (Issue B).
          "conversation" → recall_count += 1 AND last_recalled_at = now()
          "spontaneous"  → last_recalled_at = now() only
          "system"       → no reinforcement (default)

        Scoring: `_compute_final_score` のハイブリッド合成（r・t・e・a）。
        Results are re-sorted by final_score before returning.
        """
        # e 軸の基準となる今の気分を1回だけ読む（想起1回につき M は1つ）。
        # DB ロックへ入る前に読むこと：load_current_mood() は内部で db.lock を
        # 取り、これは再入不可なのでロック内から呼ぶと停止する（平均中心化 C2 で
        # 実際に起きたデッドロックと同型）。
        mood_pad: tuple[float, float, float, float] | None
        try:
            _mood = load_current_mood()
            mood_pad = (_mood.p, _mood.pn, _mood.a, _mood.dom)
        except Exception:
            logger.warning("mood を読めないので e 軸を外して想起する", exc_info=True)
            mood_pad = None

        try:
            q_vec = _coerce_to_embedding_dim(
                np.array(self._embedder.encode_query([query])[0], dtype=np.float32)
            )
            p_vec = self._get_perspective_vec(self._person_id)
            situated_q = _situated_vector(q_vec, p_vec, self._embedding_mu())
            q_sql = vec_to_sql(situated_q.tolist())

            kind_clause = "AND o.kind = %s" if kind else ""
            score_clause = "AND (1 - (s.vector <=> %s::vector)) >= %s" if min_score > 0.0 else ""
            params: list = [q_sql, self._person_id]
            if kind:
                params.append(kind)
            if min_score > 0.0:
                params += [q_sql, min_score]
            params += [q_sql, n]

            with self._db_lock:
                conn = self._ensure_connected()
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
                          {score_clause}
                        ORDER BY s.vector <=> %s::vector
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = cur.fetchall()

            if rows:
                _cfg = MemoryConfig()
                results = []
                breakdowns: dict[Any, _ScoreParts] = {}
                for row in rows:
                    cosine = float(row["score"])
                    parts = _score_breakdown(
                        cosine,
                        row["timestamp"],
                        row["last_recalled_at"],
                        int(row["recall_count"]),
                        float(row["activation_a0"]),
                        int(row["activation_n"]),
                        obs_pad=(
                            row["emotion_p"], row["emotion_pn"],
                            row["emotion_a"], row["emotion_dom"],
                        ),
                        mood_pad=mood_pad,
                        half_life_days=_cfg.recall_half_life_days,
                        floor=_cfg.recall_time_floor,
                        c_lo=_cfg.recall_c_lo,
                        c_hi=_cfg.recall_c_hi,
                        w_r=_cfg.recall_w_r,
                        w_t=_cfg.recall_w_t,
                        w_e=_cfg.recall_w_e,
                        w_a=_cfg.recall_w_a,
                        sigma=_cfg.recall_emotion_sigma,
                    )
                    final = parts.score
                    breakdowns[row["id"]] = parts
                    results.append({
                        "memory_id":        row["id"],
                        "timestamp":        row["timestamp"],
                        "summary":          row["content"],
                        "date":             _ts_to_date(row["timestamp"]),
                        "time":             _ts_to_time(row["timestamp"]),
                        "direction":        row["direction"],
                        "kind":             row["kind"],
                        "source_kind":      row["kind"],
                        "emotion":          row["emotion"],
                        "image_path":       row["image_path"],
                        "score":            final,
                        "confidence":       max(0.0, min(1.0, (cosine + 1.0) / 2.0)),
                        "retrieval_method": "semantic",
                        # mood nudge（mood-c）の入力用に PAD と activation 重みを露出。追加のみで挙動不変。
                        "emotion_pad":      MoodPAD(
                            p=row["emotion_p"], pn=row["emotion_pn"],
                            a=row["emotion_a"], dom=row["emotion_dom"],
                        ),
                        "activation":       _derive_activation(
                            float(row["activation_a0"]), int(row["activation_n"]),
                        ),
                    })
                results.sort(key=lambda r: r["score"], reverse=True)

                # 想起順の内訳。実機確認で「なぜこの順なのか」を後から再構成する
                # ために出す。記憶内容を含むうえ想起のたびに走るので debug 限定。
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "recall score: query=%r person=%s mood=%s 候補%d件",
                        query[:60], self._person_id,
                        "なし" if mood_pad is None
                        else "(%.2f,%.2f,%.2f,%.2f)" % mood_pad,
                        len(results),
                    )
                    for rank, item in enumerate(results[:10], start=1):
                        b = breakdowns[item["memory_id"]]
                        logger.debug(
                            "  #%d recall score=%.4f r=%.3f t=%.3f a=%.3f e=%s | %s | %s",
                            rank, item["score"], b.r, b.t, b.a,
                            "なし" if b.e is None else "%.3f" % b.e,
                            _ts_to_date(item["timestamp"]),
                            str(item["summary"])[:50],
                        )

                if recall_mode in ("conversation", "spontaneous"):
                    self._mark_recalled(
                        [r["memory_id"] for r in results],
                        reinforce_half_life=(recall_mode == "conversation"),
                    )
                return results

            # Fallback: plain keyword search when no situated vectors exist yet.
            # Skip when min_score > 0.0: keyword results have no cosine score so
            # the threshold cannot be enforced — return empty instead.
            if min_score > 0.0:
                return []
            return self._recall_keyword_fallback(query, n, kind)
        except Exception as e:
            logger.warning("recall failed: %s", e)
            return []

    def _recall_keyword_fallback(self, query: str, n: int, kind: str | None) -> list[dict]:
        keywords = [w for w in query.split() if len(w) > 1][:4]
        if not keywords:
            return self._recall_recency_fallback(n, kind)
        cond = " OR ".join(["o.content LIKE %s"] * len(keywords))
        params: list = [f"%{kw}%" for kw in keywords]
        kind_clause = "AND o.kind = %s" if kind else ""
        if kind:
            params.append(kind)
        params += [self._person_id, n]
        with self._db_lock:
            conn = self._ensure_connected()
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
                "date": _ts_to_date(r["timestamp"]), "time": _ts_to_time(r["timestamp"]),
                "direction": r["direction"], "kind": r["kind"],
                "source_kind": r["kind"], "emotion": r["emotion"],
                "image_path": r["image_path"],
                "confidence": 0.45, "retrieval_method": "keyword",
            }
            for r in rows
        ]

    def _recall_recency_fallback(self, n: int, kind: str | None) -> list[dict]:
        kind_clause = "AND o.kind = %s" if kind else ""
        params: list = [self._person_id]
        if kind:
            params.append(kind)
        params.append(n)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
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
                    "date": _ts_to_date(r["timestamp"]), "time": _ts_to_time(r["timestamp"]),
                    "direction": r["direction"], "kind": r["kind"],
                    "source_kind": r["kind"], "emotion": r["emotion"],
                    "image_path": r["image_path"],
                    "confidence": 0.3, "retrieval_method": "recency",
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("_recall_recency_fallback failed: %s", e); return []

    async def recall_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall, *a, **kw)

    def recent_feelings(self, n: int = 5) -> list[dict]:
        rows = self._read_observations_by_kind(
            kind=("feeling", "conversation"),
            person_id=self._person_id,
            n=n,
            columns=("content", "timestamp", "emotion"),
        )
        return [
            {"summary": r["content"], "date": _ts_to_date(r["timestamp"]),
             "time": _ts_to_time(r["timestamp"]), "emotion": r["emotion"]}
            for r in rows
        ]

    async def recent_feelings_async(self, n: int = 5):
        return await asyncio.to_thread(self.recent_feelings, n)

    def recall_self_model(self, n: int = 5) -> list[dict]:
        """Always uses AGENT_SELF_ID scope — agent's own self-understanding."""
        rows = self._read_observations_by_kind(
            kind="self_model",
            person_id=AGENT_SELF_ID,
            n=n,
            columns=("id", "content", "timestamp", "emotion", "superseded_by", "importance",
                     "emotion_p", "emotion_pn", "emotion_a", "emotion_dom"),
        )
        # A-1: 器を組み立てる経路を通す。返り値には使わず外部挙動を保つ（利用は次の一本）。
        # Y（W2a）：PAD 列を渡すことで組み立てる MI が実 PAD を emotion に載せる。返り値の
        # dict は content/timestamp/emotion しか使わないので外部挙動は不変。
        _items = [_row_to_mental_item(r) for r in rows]
        return [
            {"summary": r["content"], "date": _ts_to_date(r["timestamp"]),
             "time": _ts_to_time(r["timestamp"]), "emotion": r["emotion"]}
            for r in rows
        ]

    async def recall_self_model_async(self, n: int = 5):
        return await asyncio.to_thread(self.recall_self_model, n)

    def recall_curiosities(self, n: int = 5) -> list[dict]:
        rows = self._read_observations_by_kind(
            kind="curiosity",
            person_id=AGENT_SELF_ID,
            n=n,
            columns=("content", "timestamp"),
        )
        return [
            {"summary": r["content"], "date": _ts_to_date(r["timestamp"]),
             "time": _ts_to_time(r["timestamp"])}
            for r in rows
        ]

    async def recall_curiosities_async(self, n: int = 5):
        return await asyncio.to_thread(self.recall_curiosities, n)

    def recall_day_summaries(self, n: int = 5) -> list[dict]:
        # C-1: 所有者絞り（observations.person_id）でなく situated 相関で在席者に紐づける。
        # 母集合は所有者に依らず、この memory の person_id の視点で状況化された観測。
        rows = self._read_observations_by_situated(
            person_id=self._person_id,
            n=n,
            columns=("content", "timestamp", "emotion"),
            kind="day_summary",
        )
        return [
            {"summary": r["content"], "date": _ts_to_date(r["timestamp"]),
             "time": _ts_to_time(r["timestamp"]), "emotion": r["emotion"]}
            for r in rows
        ]

    async def recall_day_summaries_async(self, n=5):
        return await asyncio.to_thread(self.recall_day_summaries, n)

    def find_near_duplicates(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        """Return pairs of non-superseded observations whose vectors are >= threshold similar."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
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
            pairs: list[tuple[str, str, float]] = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    sim = float(vecs[i] @ vecs[j])
                    if sim >= threshold:
                        pairs.append((ids[i], ids[j], sim))
            return pairs
        except Exception as e:
            logger.warning("find_near_duplicates failed: %s", e); return []

    def create_episode(self, title: str, summary: str = "") -> str | None:
        try:
            episode_id = str(uuid.uuid4())
            now = self._now()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO episodes (id,title,summary,created_at,updated_at) "
                        "VALUES (%s,%s,%s,%s,%s)",
                        (episode_id, title, summary, now, now),
                    )
                conn.commit()
            return episode_id
        except Exception as e:
            logger.warning("create_episode failed: %s", e); return None

    def append_to_episode(self, episode_id: str, memory_id: str) -> bool:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COALESCE(MAX(position),0)+1 AS next_pos FROM episode_memories WHERE episode_id=%s",
                        (episode_id,),
                    )
                    row = cur.fetchone()
                    pos = row["next_pos"] if row else 1
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO episode_memories (id,episode_id,memory_id,position,added_at) "
                        "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (episode_id,memory_id) DO NOTHING",
                        (str(uuid.uuid4()), episode_id, memory_id, pos, self._now()),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning("append_to_episode failed: %s", e); return False

    def recall_divergent(self, query: str, n: int = 10) -> list[dict]:
        try:
            base = self.recall(query, n=n)
            if not base:
                return []
            ids = [m["memory_id"] for m in base]
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(ids))
                    cur.execute(
                        f"SELECT em.memory_id, em.episode_id, em.position "
                        f"FROM episode_memories em WHERE em.memory_id IN ({placeholders})",
                        ids,
                    )
                    ep_rows = {r["memory_id"]: r for r in cur.fetchall()}
            results = []
            for m in base:
                ep = ep_rows.get(m["memory_id"])
                results.append({
                    "memory_id": m["memory_id"],
                    "content": m.get("summary", ""),
                    "episode_id": ep["episode_id"] if ep else None,
                    "position": ep["position"] if ep else None,
                    "confidence": m.get("confidence", 0.5),
                })
            return results
        except Exception as e:
            logger.warning("recall_divergent failed: %s", e); return []

    def refresh_working_memory(self, query: str, n: int = 10) -> list[dict]:
        try:
            recalled = self.recall_divergent(query, n=n)
            if not recalled:
                return []
            now = self._now()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM memory_activation WHERE source='working_memory'")
                for item in recalled:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO memory_activation (id,memory_id,activation,source,context,episode_id,activated_at) "
                            "VALUES (%s,%s,%s,'working_memory',%s,%s,%s)",
                            (str(uuid.uuid4()), item["memory_id"], float(item.get("confidence", 0.5)),
                             query, item.get("episode_id"), now),
                        )
                conn.commit()
            return recalled
        except Exception as e:
            logger.warning("refresh_working_memory failed: %s", e); return []

    def get_working_memory(self) -> list[dict]:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT ma.memory_id,ma.activation,ma.context,ma.episode_id,ma.activated_at,"
                        "o.content FROM memory_activation ma "
                        "JOIN observations o ON o.id=ma.memory_id "
                        "WHERE ma.source='working_memory' ORDER BY ma.activation DESC",
                    )
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("get_working_memory failed: %s", e); return []

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
        pid = self._person_id
        try:
            with self._db_lock:
                conn = self._ensure_connected()
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

    def open_unfinished_business(
        self,
        summary: str,
        source: str = "agent",
        metadata: dict | None = None,
        related_memory_id: str | None = None,
    ) -> str:
        """Create an open unfinished-business record and return its ID."""
        item_id = str(uuid.uuid4())
        now = clock.now_local_iso()
        sql = (
            "INSERT INTO unfinished_business "
            "(id,summary,status,source,related_memory_id,metadata_json,created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)"
        )
        params = (
            item_id, summary, "open", source,
            related_memory_id, json.dumps(metadata or {}), now,
        )
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                conn.commit()
        except Exception as e:
            logger.warning("open_unfinished_business failed: %s", e)
        return item_id

    def list_unfinished_business(self, status: str = "open", n: int = 50) -> list[dict]:
        """Return unfinished-business records with the given status."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id,summary,status,source,created_at,metadata_json "
                        "FROM unfinished_business WHERE status=%s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (status, n),
                    )
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("list_unfinished_business failed: %s", e)
            return []

    async def list_unfinished_business_async(self, status: str = "open", n: int = 50) -> list[dict]:
        return await asyncio.to_thread(self.list_unfinished_business, status, n)

    def _get_recent_observations(self, n: int = 5) -> list[dict]:
        """Return the N most recent observations for the current person, ordered by timestamp desc."""
        try:
            today = _date.today()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, content, timestamp, emotion, direction, kind,
                               COALESCE(importance, 1.0) AS importance
                        FROM observations
                        WHERE person_id = %s AND superseded_by IS NULL
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (self._person_id, n),
                    )
                    rows = cur.fetchall()
            return [
                {
                    "memory_id":  row["id"],
                    "summary":    row["content"],
                    "date":       _ts_to_date(row["timestamp"]),
                    "time":       _ts_to_time(row["timestamp"]),
                    "emotion":    row["emotion"],
                    "direction":  row["direction"],
                    "kind":       row["kind"],
                    "importance": float(row["importance"]),
                    "confidence": float(row["importance"]),
                    "is_today":   row["timestamp"].date() == today if row["timestamp"] else False,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.debug("_get_recent_observations failed: %s", exc)
            return []

    async def as_coalition_async(self):
        """Surface recently-stored memories as a workspace coalition.

        Queries the N most recent observations by timestamp and packages them
        as a Coalition so the workspace can inject relevant context into the
        LLM prompt.  Returns None when there are no stored memories yet.
        """
        from ..workspace import Coalition

        memories = await self.recall_async("", n=5)
        if not memories:
            return None

        context = self.format_for_context(memories)
        if not context:
            return None

        confidences = [float(m.get("confidence", 0.5)) for m in memories]
        activation = min(0.6, max(confidences))

        has_today = any(m.get("is_today") for m in memories)
        novelty = 0.4 if has_today else 0.2

        top = max(memories, key=lambda m: float(m.get("confidence", 0.0)))
        summary = top["summary"]

        return Coalition(
            source="memory",
            summary=summary,
            activation=activation,
            urgency=0.1,
            novelty=novelty,
            context_block="[Memory recall]\n" + context,
        )

    # ── Format helpers (unchanged from original) ───────────────────────────

    def format_for_context(self, memories: list[dict]) -> str:
        if not memories: return ""
        lines = ["[過去の記憶（証拠つき）: conf<0.55 は不確か]:"]
        for m in memories:
            score_s = f" (類似度:{m['score']:.2f})" if "score" in m else ""
            conf = float(m.get("confidence", 0.0))
            conf_s = f" conf:{conf:.2f}"
            low = " low-confidence" if conf < 0.55 else ""
            emo  = f" [{m['emotion']}]" if m.get("emotion") and m["emotion"] != "neutral" else ""
            sid  = str(m.get("memory_id",""))[:8] or "?"
            lines.append(
                f"- {m.get('date','?')} {m.get('time','?')} id:{sid}{score_s}{conf_s}{low}"
                f" ({m.get('direction','?')}){emo}: {m['summary'][:120]}"
            )
        return "\n".join(lines)

    def format_feelings_for_context(self, f: list[dict]) -> str:
        if not f: return ""
        lines = ["[最近の気持ち・出来事]:"]
        for x in f:
            emo = f"[{x['emotion']}] " if x.get("emotion") and x["emotion"] != "neutral" else ""
            lines.append(f"- {x['date']} {x['time']} {emo}{x['summary'][:120]}")
        return "\n".join(lines)

    def format_self_model_for_context(self, sm: list[dict]) -> str:
        if not sm: return ""
        return "".join(["[うちという存在 — 経験から積み上げてきた自己像]:\n"] +
                        [f"- {m['summary'][:120]}\n" for m in sm])

    def format_curiosities_for_context(self, cs: list[dict]) -> str:
        if not cs: return ""
        return "".join(["[まだ謎のまま・続きが気になること]:\n"] +
                        [f"- {c['date']} {c['time']}: {c['summary'][:120]}\n" for c in cs])

    def format_day_summaries_for_context(self, ss: list[dict]) -> str:
        if not ss: return ""
        lines = ["[私が覚えていること — 過去の日々]:"]
        for s in ss:
            lines.append(f"- {s['date']}: {s['summary'][:200]}")
        return "\n".join(lines)


# ── MemoryTool ─────────────────────────────────────────────────────────────

class MemoryTool:
    """Agent-callable memory tools, routed through PersonMemoryManager."""

    def __init__(self, manager: "PersonMemoryManager") -> None:
        self._manager = manager
        from .pending_speech_store import PendingSpeechStore
        self._pending_store = PendingSpeechStore()
        self._notes_registered_this_turn: int = 0

    @property
    def _write_store(self) -> ObservationMemory | None:
        return self._manager.get_speaker_memory()

    @property
    def _agent_store(self) -> ObservationMemory:
        return self._manager.get_agent_memory()

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "remember",
                "description": (
                    "長期記憶に保存する。"
                    "scope: speaker=話者のみ / witnessed=その場の全員 / scene=エージェント観察 / all=全部"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content":   {"type": "string"},
                        "emotion":   {"type": "string",
                                      "enum": ["neutral","happy","sad","curious","excited","moved"]},
                        "scope":     {"type": "string",
                                      "enum": ["speaker","witnessed","scene","all"],
                                      "default": "speaker"},
                        "image_path":{"type": "string"},
                        "link_to":   {"type": "string"},
                        "link_type": {"type": "string",
                                      "enum": ["related","similar","caused_by","leads_to"]},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "recall",
                "description": "長期記憶を検索する。その場にいる全員の記憶を横断して探す。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "n":     {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "note_to_share",
                "description": (
                    "覚えた記憶を、後で誰かに話したいこととして登録する。"
                    "observation_id は remember で覚えた記憶のID。"
                    "target は話したい相手の名前(任意、省略時は誰でも)。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "observation_id": {"type": "string"},
                        "target":         {"type": "string"},
                    },
                    "required": ["observation_id"],
                },
            },
        ]

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, str | None]:
        if tool_name == "remember":
            return await self._remember(tool_input)
        if tool_name == "recall":
            return await self._recall(tool_input)
        if tool_name == "note_to_share":
            return await self._note_to_share(tool_input)
        return f"Unknown memory tool: {tool_name}", None

    async def _note_to_share(self, inp: dict) -> tuple[str, None]:
        obs_id = inp["observation_id"]
        target_name = inp.get("target")
        target_id: str | None = None
        if target_name:
            target_id = self._manager.find_person_id_by_name(target_name)
            # 名前解決失敗 → NULL(誰でも)にフォールバック
        pid = self._pending_store.add(obs_id, target_id)
        if pid is None:
            return "覚えていない記憶は話せません。先に remember してください。", None
        self._notes_registered_this_turn = getattr(self, "_notes_registered_this_turn", 0) + 1
        return "話したいこととして登録しました。", None

    async def _remember(self, inp: dict) -> tuple[str, None]:
        scope      = inp.get("scope", "speaker")
        content    = inp["content"]
        emotion    = inp.get("emotion", "neutral")
        image_path = inp.get("image_path")
        link_to    = inp.get("link_to")
        link_type  = inp.get("link_type", "related")

        present_ids = self._manager.get_present_ids()
        speaker_id  = self._manager.current_speaker_id
        results: list[str] = []

        # speaker
        if scope in ("speaker", "all"):
            store = self._write_store
            if store is None:
                return "話者不明のため書き込めません。declare_speaker を先に呼んでください。", None
            mem_id, ok = await store.save_async_with_id(
                content, kind="utterance", emotion=emotion,
                image_path=image_path,
                writer_id=speaker_id, subject_id=speaker_id,
                participants=present_ids, scope="speaker",
            )
            if ok:
                results.append(f"[{self._manager.get_person_name(speaker_id)}] 話者")
                if link_to and mem_id:
                    await store.link_memories_async(mem_id, link_to, link_type=link_type)

        # witnessed (listeners)
        if scope in ("witnessed", "all"):
            sp_name = self._manager.get_person_name(speaker_id) if speaker_id else "?"
            for pid, mem in self._manager.get_all_present_memories():
                if pid == speaker_id:
                    continue
                witnessed = f"[{sp_name}が言った] {content}"
                await mem.save_async(
                    witnessed, kind="witnessed", emotion=emotion,
                    writer_id=pid, subject_id=speaker_id,
                    participants=present_ids, scope="witnessed",
                )
            listeners = [self._manager.get_person_name(p)
                         for p, _ in self._manager.get_all_present_memories()
                         if p != speaker_id]
            if listeners:
                results.append(f"[{', '.join(listeners)}] 目撃")

        # scene
        if scope in ("scene", "all"):
            sp_name   = self._manager.get_person_name(speaker_id) if speaker_id else "不明"
            pnames    = [self._manager.get_person_name(p) for p in present_ids]
            scene_txt = f"[場面] 参加者: {', '.join(pnames)} / 発言者: {sp_name} / {content}"
            await self._agent_store.save_async(
                scene_txt, kind="scene", emotion=emotion,
                participants=present_ids, scope="scene",
                writer_id=AGENT_SELF_ID,
            )
            results.append("[agent_self] 場面記録")

        summary = " / ".join(results) if results else "書き込みなし"
        return f"記憶しました: {summary}", None

    async def _recall(self, inp: dict) -> tuple[str, None]:
        query = inp["query"]
        n     = int(inp.get("n", 3))
        all_results: list[dict] = []

        # agent self
        agent_mem = self._agent_store
        for m in await agent_mem.recall_async(query, n=n, recall_mode="conversation"):
            m["_from"] = "自分"
            all_results.append(m)

        # all present persons
        for pid, mem in self._manager.get_all_present_memories():
            name = self._manager.get_person_name(pid)
            for m in await mem.recall_async(query, n=n, recall_mode="conversation"):
                m["_from"] = name
                all_results.append(m)

        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top = all_results[:n]

        if not top:
            return "関連する記憶はありません。", None

        lines = []
        for m in top:
            s = f" ({m['score']:.2f})" if "score" in m else ""
            lines.append(
                f"[{m.get('_from','?')}] {m['date']} {m['time']}{s} [{m['emotion']}]: {m['summary'][:130]}"
            )
        return "\n".join(lines), None
