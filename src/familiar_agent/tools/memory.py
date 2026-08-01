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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import MemoryConfig, RecallWeights
from ..db import Database, get_db, vec_to_sql
from ..mood_register import MoodPAD, load_current_mood
from ..core.mental_item import (  # noqa: F401  既存の呼び出し側が memory 経由で引くための再輸出
    MentalItem,
    PrimitiveMentalItem,
    _row_to_mental_item,
)
from ..db_migrations import apply_migrations, default_migration_dir
from ..legacy.semantic_layer import LegacySemanticLayer
from ..store import clock
from ..store.context import StoreContext
from ..store.jobs import JobQueue
from ..store.persons import PersonRegistry
from ..store.observations import ObservationStore
from ..store.situated import (  # noqa: F401  既存の呼び出し側が memory 経由で引くための再輸出
    SituatedVectors,
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
    """想起の採点の内訳。`e` は項を外したとき None。

    2つの合成量がある。**地力（merit・記号 `m`）**＝加算部の加重平均で、時間経過を
    含んだ重要度。関連は入らない。**適合度（fit・記号 `f`）**＝`r^(w_r) × m` で、
    W を選ぶのはこれ。いまの問いへの適合を含むので、同じ記録でも問いが変われば変わる。
    """

    fit: float
    r: float
    t: float
    g: float
    m: float
    e: float | None
    p: float | None = None


def _score_breakdown(
    cosine: float,
    ts,
    last_recalled_at,
    recall_count: int,
    groundedness_g0: float,
    groundedness_n: int,
    *,
    obs_pad: tuple[float, float, float, float] | None = None,
    mood_pad: tuple[float, float, float, float] | None = None,
    half_life_days: float,
    floor: float,
    reference_epoch: float | None = None,
    c_lo: float = 0.0,
    c_hi: float = 1.0,
    w_r: float = 1.0,
    w_t: float = 1.0,
    w_e: float = 1.0,
    w_g: float = 1.5,
    w_p: float = 0.0,
    p: float | None = None,
    sigma: float = 1.0,
    g_floor: float = 0.0,
) -> _ScoreParts:
    """想起スコアと各軸の内訳を返す（合成式の正本）。

    `_compute_final_score` はここから `score` を取り出すだけの薄い包みである。
    内訳を別に組み立てると式が二重になり、片方だけ直したときにずれるので、
    計算はこの関数だけが持つ。`e` は項を外したとき None になる。

    式と各軸の意味は `_compute_final_score` の docstring にある。
    """
    # 起点は**書かれた時刻**だけ。強化B（使ったら若返る）は仕組みごと後回しにした。
    # `last_recalled_at` を起点にすると、想起で更新されるたびに t=1 へ戻り、一度上がった
    # 記録が自分を押し上げ続ける（実機で 47日前の挨拶が t=1.000 で居座り、5秒前の自分の
    # 発話を押し出した）。設計（課題5 F節）も「想起では触らない」と定めている。
    base = ts
    origin = _to_epoch(base)
    if origin is None:
        origin = datetime.now(timezone.utc).timestamp()
    state = DecayState(
        origin_epoch=origin,
        half_life_seconds=half_life_days * 86400.0,
        floor=floor,
        # **強化A（想起回数で実効半減期を伸ばす）は使わない。** `課題5` F節が廃止と
        # 確定させている（重要さは根づきの n が担い、t は純粋な時間減衰）。
        # 残っていたため、`recall_count` が 20 なら半減期が 3×2^20 日＝8600年になり、
        # **何度も想起された古い記録が永久に t=1** になっていた。実機で 47日前の挨拶が
        # t=1.000 で上位を占め、5秒前の自分の発話を押し出した（「おかえりなさい」を2回）。
    )
    # 時間軸の基準。既定は「いま」だが、調停が人の言葉から動かせる（「去年の夏の話」）。
    ref_epoch = (reference_epoch if reference_epoch is not None
                 else datetime.now(timezone.utc).timestamp())

    r = _stretch_relevance(cosine, c_lo=c_lo, c_hi=c_hi)
    t = state.score(ref_epoch)
    # open な記録は下限で持ち上げる（a_open・`課題5_パラメータ仮案` §184）。max なので
    # 導出が下限より高い記録は下がらない。open でない記録は g_floor=0.0 で素通りする。
    g = max(_derive_groundedness(float(groundedness_g0), int(groundedness_n)), g_floor)

    e: float | None = None
    numerator = w_t * t + w_g * g
    denominator = w_t + w_g
    if obs_pad is not None and mood_pad is not None:
        e = _emotion_match(obs_pad, mood_pad, sigma=sigma)
        numerator += w_e * e
        denominator += w_e

    # 在席者相関 p（第5軸・役割2）。在席他者ゼロなら p は None で項ごと外す。
    if p is not None and w_p > 0.0:
        numerator += w_p * p
        denominator += w_p

    m = 1.0 if denominator <= 0.0 else numerator / denominator
    return _ScoreParts(fit=(r ** w_r) * m, r=r, t=t, g=g, m=m, e=e, p=p)


def _compute_final_score(
    cosine: float,
    ts,
    last_recalled_at,
    recall_count: int,
    groundedness_g0: float,
    groundedness_n: int,
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
    w_g: float = 1.5,
    sigma: float = 1.0,
) -> float:
    """想起スコアのハイブリッド合成（課題5 v0.24 D 節・Phase 2 スライス3）。

        score = r^{w_r} × M,  M = (w_t·t + w_e·e + w_g·a) / (w_t + w_e + w_g)

    関連 r だけが乗算ゲート（段階的関連係数）で、t・e・a は加重平均で補償的に
    束ねる。純積ではないので、一軸が低いだけで記憶が消えることはない。加算部の
    重みが全0なら M=1（score は r だけ）。

    - r：`_stretch_relevance` でコサインを伸長した値。
    - t：`DecayState`（強化B＝last_recalled_at を origin にして若返り。**強化A＝
      recall_count で半減期を伸ばす仕組みは使わない**＝`課題5` F節で廃止確定。
      重要さは根づきの n が担い、t は純粋な時間減衰）。時間減衰はこの軸に一元化し、
      importance の日次減衰は使わない（P-1・[D-想起合成]）。
    - e：`_emotion_match(obs_pad, mood_pad)`＝**今の気分と観測 PAD の距離**。
      記憶どうしの感情距離ではない（`感情ループ全体像` の `M → RECALL`）。
      mood_pad が None（mood を読めなかった経路）のときは e 項を分子分母から
      外す。中立0.5で埋めると「気分に一致する記憶」を偽って作ってしまうため。
      obs_pad が None のときも同様に外す。
    - a：`_derive_groundedness(a0, n)`（イベント駆動・時間では減らさない）。
    - p（在席者相関）は知覚待ちのため項ごと持たない。課題5 の「在席者ゼロなら
      w_p 項を外す」に一致する。

    係数は MemoryConfig から注入する。計算の実体は `_score_breakdown` にある
    （内訳ログと式を共有するため）。
    """
    return _score_breakdown(
        cosine, ts, last_recalled_at, recall_count, groundedness_g0, groundedness_n,
        obs_pad=obs_pad, mood_pad=mood_pad,
        half_life_days=half_life_days, floor=floor,
        c_lo=c_lo, c_hi=c_hi,
        w_r=w_r, w_t=w_t, w_e=w_e, w_g=w_g, sigma=sigma,
    ).fit


# ── Helpers ────────────────────────────────────────────────────────────────

# 時刻の整形は store/clock.py が持つ。既存の呼び出し名はそのまま使えるようにする。
_ts_to_date = clock.ts_to_date
_ts_to_time = clock.ts_to_time






def _derive_groundedness(
    a0: float, n: int, *, floor: float = 0.0, c: float = 2.0,
    epsilon: float = 0.001, step: float = 0.33,
) -> float:
    """初期値 a0 と正味デルタ回数 n から活性 a を導出する。

    a0 を [floor, c] で正規化し、ε で両端に寄せてロジットで無限区間へ写し、
    n·step を足してロジスティックで [floor, c] へ戻す。n=0 なら g=g0。
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

class ObservationMemory:
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
        self._build_layers()
        import os
        if os.environ.get("FAMILIAR_EMBEDDING_PREWARM", "1").lower() not in {"0","false","no","off"}:
            self._embedder.pre_warm()

    def _build_layers(self) -> None:
        """層を組み立てる。依存は引数に出す（宿主の名前空間を共有しない）。

        向きは jobs → observations → situated / legacy の一方向。
        """
        self._ctx = StoreContext(
            db=self._db, lock=self._db_lock,
            person_id=self._person_id, embedder=self._embedder,
        )
        self._situated = SituatedVectors(self._ctx)
        self._legacy = LegacySemanticLayer(self._ctx)
        self._observations = ObservationStore(
            self._ctx, situated=self._situated, legacy=self._legacy
        )
        self._jobs = JobQueue(self._ctx, observations=self._observations)
        self._persons_store = PersonRegistry(self._ctx)

    # ── Internals ──────────────────────────────────────────────────────────

    def is_embedding_ready(self) -> bool:
        return self._embedder.is_ready()

    def embedding_failed(self) -> bool:
        return self._embedder.failed()

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
        return clock.now_utc_iso()


    # ── 層への委譲 ────────────────────────────────────────────────────────
    # 公開面は従来どおり ObservationMemory に集める（呼び出し側を変えないため）。
    # 委譲するのは**外から呼ばれるものだけ**で、層の内部ヘルパーは委譲しない
    # （層の内側が外から触れる状態を残さないため）。署名は層の実物に合わせて
    # 書く。`*a, **kw` にすると、何を受けるかがここから読めなくなる。
    # 自動転送（__getattr__）は使わない。何が公開面かがコードから読めなくなる。

    # 人物レジストリ（store/persons.py）
    def register_person(self, name: str, display_name: str = "", person_id: str | None = None) -> str:
        return self._persons_store.register_person(name, display_name, person_id)

    def list_persons(self) -> list[dict]:
        return self._persons_store.list_persons()

    # 観測（store/observations.py）
    def mark_superseded(self, old_id: 'str', new_id: 'str') -> 'None':
        return self._observations.mark_superseded(old_id, new_id)

    def apply_verdicts(self, verdicts: dict[str, str]) -> int:
        """想起した記憶の扱いの申告を反映する（store 層へ委譲）。"""
        return self._observations.apply_verdicts(verdicts)

    # 発話の記録へ足す印。**一度だけ**足す。何を調べているかは求めの版が持つので、
    # 発話の記録が持つ必要はない（二重に持つと、どちらが正かが曖昧になる）。
    LOOKUP_STARTED_NOTE = "検索を始めた"

    def note_lookup_started(self, obs_id: str) -> bool:
        """発話の記録へ「検索を始めた」を一度だけ足し、埋め込みを作り直す。

        発話の記録は求めの版チェーンの外にあるが、検索を始めたことはそこからも辿れた
        ほうがよい。足したときだけ True を返す。
        """
        if not obs_id:
            return False
        try:
            return self._observations.append_and_reembed(obs_id, self.LOOKUP_STARTED_NOTE)
        except Exception:
            logger.exception("note_lookup_started failed: %.8s", obs_id)
            return False

    def decay_importance(self, before_date: 'str', factor: 'float' = 0.95) -> 'int':
        return self._observations.decay_importance(before_date, factor)

    async def decay_importance_async(self, *a, **kw):
        return await self._observations.decay_importance_async(*a, **kw)

    def get_dates_with_observations(self, days: 'int' = 7) -> 'list[str]':
        return self._observations.get_dates_with_observations(days)

    def get_dates_with_summaries(self) -> 'list[str]':
        return self._observations.get_dates_with_summaries()

    def get_observations_for_date(self, date: 'str', limit: 'int' = 50) -> 'list[dict]':
        return self._observations.get_observations_for_date(date, limit)

    def delete_day_summaries_for_date(self, date: 'str') -> 'int':
        return self._observations.delete_day_summaries_for_date(date)

    def recall_on_this_day(self, month: 'int', day: 'int', n: 'int' = 5) -> 'list[dict]':
        return self._observations.recall_on_this_day(month, day, n)

    async def recall_on_this_day_async(self, month: 'int', day: 'int', n: 'int' = 5) -> 'list[dict]':
        return await self._observations.recall_on_this_day_async(month, day, n)

    def get_earliest_date(self) -> 'str | None':
        return self._observations.get_earliest_date()

    async def get_earliest_date_async(self) -> 'str | None':
        return await self._observations.get_earliest_date_async()


    def find_near_duplicates(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        return self._observations.find_near_duplicates(threshold)

    def pick_seed_candidates(
        self, hour: int, month: int, *, hour_window: int, month_window: int, k: int
    ) -> list[dict]:
        return self._observations.pick_seed_candidates(
            hour, month, hour_window=hour_window, month_window=month_window, k=k
        )

    # キュー（store/jobs.py）
    def append_memory_event(self, event_type: 'str', payload: 'dict', dedupe_key: 'str | None' = None, queue_job: 'bool' = True, job_type: 'str' = 'materialize_observation') -> 'tuple[str | None, bool]':
        return self._jobs.append_memory_event(event_type, payload, dedupe_key, queue_job, job_type)

    async def append_memory_event_async(self, *a, **kw):
        return await self._jobs.append_memory_event_async(*a, **kw)

    def claim_pending_jobs(self, limit: 'int' = 10) -> 'list[dict]':
        return self._jobs.claim_pending_jobs(limit)

    def mark_job_done(self, job_id: 'str') -> 'bool':
        return self._jobs.mark_job_done(job_id)

    def mark_job_failed(self, job_id: 'str', error: 'str', retry_delay: 'float' = 10.0, max_attempts: 'int' = 3) -> 'str':
        return self._jobs.mark_job_failed(job_id, error, retry_delay, max_attempts)

    def materialize_event(self, event_id: 'str') -> 'bool':
        return self._jobs.materialize_event(event_id)


    # 撤去予定の層（legacy/semantic_layer.py・Phase 6 で消える）
    def recall_semantic_facts(self, query: 'str', n: 'int' = 5) -> 'list[dict]':
        return self._legacy.recall_semantic_facts(query, n)

    async def recall_semantic_facts_async(self, *a, **kw):
        return await self._legacy.recall_semantic_facts_async(*a, **kw)

    def recall_behavior_policies(self, query: 'str', n: 'int' = 5) -> 'list[dict]':
        return self._legacy.recall_behavior_policies(query, n)

    async def recall_behavior_policies_async(self, *a, **kw):
        return await self._legacy.recall_behavior_policies_async(*a, **kw)

    def recall_revisions(self, entity_type: 'str' = 'semantic_fact', entity_key: 'str | None' = None, n: 'int' = 50) -> 'list[dict]':
        return self._legacy.recall_revisions(entity_type, entity_key, n)

    def adjust_semantic_fact_confidence(self, key: 'str', delta: 'float', reason: 'str' = ''):
        return self._legacy.adjust_semantic_fact_confidence(key, delta, reason)

    async def adjust_semantic_fact_confidence_async(self, key: 'str', delta: 'float', reason: 'str' = ''):
        return await self._legacy.adjust_semantic_fact_confidence_async(key, delta, reason)

    def adjust_behavior_policy_confidence(self, key: 'str', delta: 'float', reason: 'str' = ''):
        return self._legacy.adjust_behavior_policy_confidence(key, delta, reason)

    async def adjust_behavior_policy_confidence_async(self, key: 'str', delta: 'float', reason: 'str' = '', policy_text: 'str' = '', trigger_context: 'str' = '', action_hint: 'str' = ''):
        return await self._legacy.adjust_behavior_policy_confidence_async(key, delta, reason, policy_text, trigger_context, action_hint)

    def link_memories(self, src: 'str', tgt: 'str', link_type: 'str' = 'related', note: 'str | None' = None) -> 'bool':
        return self._legacy.link_memories(src, tgt, link_type, note)

    async def link_memories_async(self, *a, **kw):
        return await self._legacy.link_memories_async(*a, **kw)

    def get_linked_memories(self, memory_id: 'str', direction: 'str' = 'both') -> 'list[dict]':
        return self._legacy.get_linked_memories(memory_id, direction)

    async def get_linked_memories_async(self, *a, **kw):
        return await self._legacy.get_linked_memories_async(*a, **kw)

    def format_semantic_facts_for_context(self, facts: 'list[dict]') -> 'str':
        return self._legacy.format_semantic_facts_for_context(facts)

    def format_behavior_policies_for_context(self, policies: 'list[dict]') -> 'str':
        return self._legacy.format_behavior_policies_for_context(policies)

    # ── Person management ──────────────────────────────────────────────────

    def for_person(self, person_id: str) -> "ObservationMemory":
        """Return a lightweight view of this memory scoped to another person."""
        obj = object.__new__(ObservationMemory)
        obj._person_id = person_id
        obj._db        = self._db
        obj._db_lock   = self._db_lock
        obj._embedder  = self._embedder
        obj._build_layers()
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
                # 直接 save へフォールバックする回復経路。trace は残す。
                logger.warning("append_memory_event failed, continuing with direct save: %s", e, exc_info=True)
            obs_id = event_id or str(uuid.uuid4())
            _cfg = MemoryConfig()
            return bool(self._observations.materialize_save_event(
                obs_id, payload,
                dedup_window_secs=_cfg.dedup_window_secs,
                writer_id=writer_id,
                subject_id=subject_id,
                participants=participants,
                scope=scope,
                novelty_k=_cfg.novelty_k,
                novelty_w_n=_cfg.novelty_w_n,
                novelty_default=_cfg.novelty_default,
                novelty_a0_cap=_cfg.novelty_a0_cap,
            ))
        except Exception:
            # 保存の失敗（埋め込み次元不一致・モデル未ロード・コードバグ等の決定的
            # エラーを含む）はトレース付きで loud に残す。返りは False（ターンは落とさない）。
            logger.exception("save failed")
            return False

    def save_with_id(self, content: str, **kwargs) -> tuple[str | None, bool]:
        _pad = kwargs.get("emotion_pad")
        payload = dict(content=content, direction=kwargs.get("direction","unknown"),
                       kind=kwargs.get("kind","observation"), emotion=kwargs.get("emotion","neutral"),
                       image_path=kwargs.get("image_path"), override_date=kwargs.get("override_date"),
                       emotion_pad=_pad.to_json_dict() if _pad else None,
                       parent_id=kwargs.get("parent_id"))
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
            # 返るのは「この内容を保持する行の id」。重複スキップなら既存行の id なので、
            # 呼び出し側は必ず実在する行を指す（supersede の宛先に使える）。
            stored_id = self._observations.materialize_save_event(
                obs_id, payload,
                dedup_window_secs=MemoryConfig().dedup_window_secs,
                writer_id=kwargs.get("writer_id"),
                subject_id=kwargs.get("subject_id"),
                participants=kwargs.get("participants"),
                scope=kwargs.get("scope", "speaker"),
            )
            return stored_id, stored_id is not None
        except Exception:
            logger.exception("save_with_id failed")
            return None, False

    async def save_async(self, *a, **kw) -> bool:
        return await asyncio.to_thread(self.save, *a, **kw)

    async def save_async_with_id(self, *a, **kw) -> tuple[str | None, bool]:
        return await asyncio.to_thread(self.save_with_id, *a, **kw)

    def content_novelty(self, content: str) -> float:
        """内容の新規性 novelty ∈ [0,1]（AGENT_SELF 視点・self_model 除外・課題5 v0.26）。

        a0（保存時の活性）と A（評価器 arousal）の源。空内容は既定を返す。埋め込みは
        重いのでロック外で計算し、近傍検索だけロック内で行う。
        """
        cfg = MemoryConfig()
        content = (content or "").strip()
        if not content:
            return cfg.novelty_default
        try:
            vec = self._embedder.encode_document([content])[0]
        except Exception as e:  # noqa: BLE001
            logger.warning("content_novelty embed failed, using default: %s", e)
            return cfg.novelty_default
        with self._db_lock:
            conn = self._ctx.conn()
            return self._observations.content_novelty(
                np.asarray(vec, dtype=np.float32), conn,
                k=cfg.novelty_k, default=cfg.novelty_default,
            )

    async def content_novelty_async(self, content: str) -> float:
        return await asyncio.to_thread(self.content_novelty, content)

    def _presence_correlation(
        self, q_vec, obs_ids: list[str], present_others: list[str],
        *, c_lo: float, c_hi: float,
    ) -> dict[str, float]:
        """在席者相関 p（obs_id → [0,1]・課題5 v0.26／[D-在席相関]）。

        在席他者 q ごとに、クエリを q 視点で situate した situated コサインを r と同じ
        伸長で r_{p,q} 化し、obs_id ごとに noisy-OR `p = 1 − Π_q(1 − r_{p,q})` で束ねる。
        自分（AGENT_SELF）・話者の除外は呼び出し側で行う（present_others に含めない）。
        present_others か obs_ids が空なら {}（在席他者ゼロ＝p 項を外す）。
        """
        if not obs_ids or not present_others:
            return {}
        mu = self._situated._embedding_mu()
        one_minus: dict[str, float] = {oid: 1.0 for oid in obs_ids}
        for q in present_others:
            p_vec_q = self._situated._get_perspective_vec(q)
            sit_q = _situated_vector(q_vec, p_vec_q, mu)
            cosines = self._observations.situated_cosines(
                vec_to_sql(sit_q.tolist()), list(obs_ids), q,
            )
            for oid, cos in cosines.items():
                r_pq = _stretch_relevance(cos, c_lo=c_lo, c_hi=c_hi)
                one_minus[oid] *= (1.0 - r_pq)
        return {oid: 1.0 - om for oid, om in one_minus.items()}

    def _diffuse_extend(self, results: list[dict], cfg, seed_vec=None) -> list[dict]:
        """拡散想起 (A)共起＋(B)主体で W を有界再帰で広げ、a0=0 の W 要素（dict）を返す。

        seed_vec があれば候補を seed から遠い順（新規性高い順）に並べ替えて novel を優先する（4b）。
        """
        try:
            from ..core.diffuse import diffuse_ids, select_entity_seeds
            from ..diffuse_store import (
                cooccurring_mi_ids,
                fetch_diffuse_rows,
                fetch_perspectives,
                order_ids_by_farthest,
                recall_by_person,
            )
            from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

            seed_ids = [r["memory_id"] for r in results]
            exclude = {AGENT_SELF_ID, DEFAULT_PERSON_ID, self._person_id}
            cap = max(1, cfg.diffuse_max_add)
            with self._db_lock:
                conn = self._db.conn()

                def _get_candidates(known: list[str]) -> list[str]:
                    cands = list(cooccurring_mi_ids(conn, known, min_shared=2, limit=cap * 4))
                    for pid in select_entity_seeds(fetch_perspectives(conn, known), exclude)[:cap]:
                        cands += recall_by_person(conn, pid, limit=cap)
                    # 4b：seed から遠い順（新規性高い順）に並べ替えて novel を優先する。
                    if seed_vec is not None:
                        cands = order_ids_by_farthest(conn, cands, seed_vec)
                    return cands

                added = diffuse_ids(
                    seed_ids, _get_candidates,
                    max_add=cap, max_depth=max(1, cfg.diffuse_max_depth),
                )
                extra = fetch_diffuse_rows(conn, added)
            if extra:
                logger.info("diffuse recall: +%d MI（(A)共起/(B)主体）", len(extra))
            return extra
        except Exception:
            logger.warning("diffuse recall failed", exc_info=True)
            return []

    def recall(self, query: str, n: int = 3, kind: str | None = None,
               min_score: float = 0.0,
               present_others: list[str] | None = None,
               exclude_ids: list[str] | None = None,
               time_ref: float | None = None,
               time_span_days: float | None = None,
               weights: "RecallWeights | None" = None,
               open_ids: list[str] | None = None) -> list[dict]:
        """Recall using situated vectors (pgvector cosine search).

        min_score:   合成 final score の soft 床（生コサインではない）。無関係の
                     最終排除を担う（根拠台帳 §3–4）。候補をいくつ集めるかは床とは別で、
                     一次絞り件数 N（`recall_primary_n`）が決める。

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
            p_vec = self._situated._get_perspective_vec(self._person_id)
            situated_q = _situated_vector(q_vec, p_vec, self._situated._embedding_mu())
            q_sql = vec_to_sql(situated_q.tolist())

            _cfg = MemoryConfig()
            # 呼び出し側が trigger 別の採用値を渡してくる。渡さない経路（連想想起など）は
            # 基底のプロファイルで採点する（挙動不変）。
            _w = weights or RecallWeights(
                _cfg.recall_w_r, _cfg.recall_w_t, _cfg.recall_w_e,
                _cfg.recall_w_g, _cfg.recall_w_p,
            )
            # open な記録（この求めのために書いた、まだ決着していない O）。活性に下限を
            # 課して W へ浮かせる。集合にするのは行ごとに引くため。
            _open = set(open_ids or ())
            # 一次絞り件数 N（軸あたり・Config `recall_primary_n`）。各軸はここで N 件集め、
            # 採点して上位 n 件（正本の W 載せ上限 K）へ絞る。N は再スコア用でフルLLM へは
            # 渡らないので、トークン量とは無関係に広く取れる（`課題5_パラメータ仮案` §93）。
            # 床（min_score）の有無で N を変えない。床は採点後に効く別の決定である。
            fetch_n = _cfg.recall_primary_n
            speaker_rows = self._observations.by_vector(
                q_sql, fetch_n, kind=kind, exclude_ids=exclude_ids
            )

            # 候補の obs_id → 行（列は obs レベルなのでどの視点由来でも同じ）。
            row_by_id: dict[str, dict] = {r["id"]: r for r in speaker_rows}
            # r（関連）の素点＝話者視点 situated コサイン。話者候補はそのまま持っている。
            cos_by_id: dict[str, float] = {r["id"]: float(r["score"]) for r in speaker_rows}

            # 新しさ軸の一次絞り（[D-想起合成] の多軸 union）。関連軸だけで候補を作ると、
            # 話題が近くない限り直近の記録が候補にすら入らず、t 軸は並べ替えにしか
            # 効かない。直前の会話を「思い出せない」のはこれが理由だった（実機で観測）。
            # w_t=0 のプロファイルではこの軸を集めない（設計どおり重み>0 の軸だけ union）。
            # 時間軸の一次絞り。基準時刻からの隔たりで取る（既定の基準は「いま」）。
            # 幅の指定があるときは、書かれた時刻と使った時刻の両方で探す。
            ref_epoch = time_ref if time_ref is not None else datetime.now(timezone.utc).timestamp()
            if _cfg.recall_w_t > 0.0:
                for r in self._observations.by_time(
                    ref_epoch, fetch_n, span=time_span_days is not None,
                    kind=kind, exclude_ids=exclude_ids,
                ):
                    row_by_id.setdefault(r["id"], r)

            # 在席者相関 p（第5軸・役割2）の候補集合拡張（slice-2）。在席他者 q 視点でも
            # 候補を取って union し、話者の問いと無関係でも在席他者に結びつく記憶を W に上げる。
            # トグルで slice-1（話者候補の再採点のみ）へ退避できる。
            if present_others and _cfg.recall_presence_expand:
                mu = self._situated._embedding_mu()
                for q in present_others:
                    sit_q = _situated_vector(q_vec, self._situated._get_perspective_vec(q), mu)
                    for r in self._observations.by_vector(
                        vec_to_sql(sit_q.tolist()), fetch_n, kind=kind,
                        exclude_ids=exclude_ids,
                    ):
                        row_by_id.setdefault(r["id"], r)  # 新規候補だけ足す
                # 在席他者由来で話者候補に無い記憶へ、話者視点の r を補って公平に採点する。
                extra = [oid for oid in row_by_id if oid not in cos_by_id]
                if extra:
                    cos_by_id.update(
                        self._observations.situated_cosines(q_sql, extra, self._person_id)
                    )

            # 感情軸の一次絞り。出発点は**そのターンの気分**で、気分が動けば候補も変わる。
            # 距離はロジット空間の L2（√λ 畳み込み済み）＝採点の D と同じもの。
            if _cfg.recall_w_e > 0.0 and mood_pad is not None:
                from ..emotion_pad import pad_to_search_vector

                mood_vec = "[" + ",".join(
                    f"{v:.6f}" for v in pad_to_search_vector(mood_pad)) + "]"
                for r in self._observations.by_emotion(
                    mood_vec, fetch_n, kind=kind, exclude_ids=exclude_ids
                ):
                    row_by_id.setdefault(r["id"], r)

            # 関連軸以外から入った候補には score 列が無いので、話者視点の r を補って
            # 公平に採点する（在席者相関の拡張と同じやり方）。
            missing = [oid for oid in row_by_id if oid not in cos_by_id]
            if missing:
                cos_by_id.update(
                    self._observations.situated_cosines(q_sql, missing, self._person_id)
                )

            # p は union 全体に対して計算。在席他者ゼロなら空＝各行 p=None で項落ち（不変）。
            p_by_id: dict[str, float] = {}
            if present_others:
                p_by_id = self._presence_correlation(
                    q_vec, list(row_by_id), present_others,
                    c_lo=_cfg.recall_c_lo, c_hi=_cfg.recall_c_hi,
                )

            rows = list(row_by_id.values())
            if rows:
                results = []
                breakdowns: dict[Any, _ScoreParts] = {}
                for row in rows:
                    cosine = cos_by_id.get(row["id"], 0.0)
                    parts = _score_breakdown(
                        cosine,
                        row["timestamp"],
                        row["last_recalled_at"],
                        int(row["recall_count"]),
                        float(row["groundedness_g0"]),
                        int(row["groundedness_n"]),
                        obs_pad=(
                            row["emotion_p"], row["emotion_pn"],
                            row["emotion_a"], row["emotion_dom"],
                        ),
                        mood_pad=mood_pad,
                        half_life_days=(time_span_days
                                        if time_span_days is not None
                                        else _cfg.recall_half_life_days),
                        reference_epoch=ref_epoch,
                        floor=_cfg.recall_time_floor,
                        c_lo=_cfg.recall_c_lo,
                        c_hi=_cfg.recall_c_hi,
                        w_r=_w.w_r,
                        w_t=_w.w_t,
                        w_e=_w.w_e,
                        w_g=_w.w_g,
                        w_p=_w.w_p,
                        p=p_by_id.get(row["id"]),
                        sigma=_cfg.recall_emotion_sigma,
                        g_floor=(_cfg.recall_g_open if row["id"] in _open else 0.0),
                    )
                    final = parts.fit
                    # 合成スコアの soft 床。生コサインではなく最終スコアで絞る。
                    if min_score > 0.0 and final < min_score:
                        continue
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
                        "fit":              final,
                        "confidence":       max(0.0, min(1.0, (cosine + 1.0) / 2.0)),
                        "retrieval_method": "semantic",
                        # mood nudge（mood-c）の入力用に PAD と根づきの重みを露出。追加のみで挙動不変。
                        # **ここには a_open の下限を掛けない（意図的）。** この値は
                        # `compute_n_pad` で N_PAD の加重平均の**重み**になる。実測の
                        # 根づきは 0.057 程度（根拠台帳）で、open な記録を 1.0 で
                        # 入れると調査中の mood をトリガ O がほぼ独占し、人の問い1件の
                        # PAD が気分を決めてしまう。下限は「W から落とさない」ための
                        # ものなので、採点に使う a にだけ効かせる。
                        "emotion_pad":      MoodPAD(
                            p=row["emotion_p"], pn=row["emotion_pn"],
                            a=row["emotion_a"], dom=row["emotion_dom"],
                        ),
                        "groundedness":       _derive_groundedness(
                            float(row["groundedness_g0"]), int(row["groundedness_n"]),
                        ),
                    })
                results.sort(key=lambda r: r["fit"], reverse=True)
                # 一次絞りで集めた N 件から、上位 n 件（正本の W 載せ上限 K）へ絞る。
                results = results[:n]

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
                            rank, item["fit"], b.r, b.t, b.g,
                            "なし" if b.e is None else "%.3f" % b.e,
                            _ts_to_date(item["timestamp"]),
                            str(item["summary"])[:50],
                        )

                # **想起では強化しない。** 更新すべきは「フルLLM が実際に参照した MI」
                # だけ（課題5 F節・強化B「想起では触らない」）だが、その判定は未実装なので
                # 仕組みごと後回しにした。想起しただけで若返らせると、一度上がった記録が
                # 自分を押し上げ続ける（実機で 47日前の挨拶が t=1.000 で居座った）。
                # store 側の `_mark_recalled` は判定ができたときのために残す。

                # 拡散想起（[D-WR拡散想起]・4a）：(A)共起＋(B)主体で W を再帰的に広げ、
                # g0=0（適合度も根づきも 0）で末尾へ足す（top-n の後・reinforce しない＝DB 非破壊）。
                if _cfg.diffuse_recall and results:
                    results.extend(self._diffuse_extend(results, _cfg, seed_vec=q_vec))

                return results

            # Fallback: plain keyword search when no situated vectors exist yet.
            # Skip when min_score > 0.0: keyword results have no cosine score so
            # the threshold cannot be enforced — return empty instead.
            if min_score > 0.0:
                return []
            return self._observations.keyword_fallback(query, n, kind)
        except (TypeError, AttributeError, NameError):
            # コードの誤り（署名不一致・属性ミスなど）は degrade しない。`[]` に化けると
            # 「想起0件」に見えて原因が隠れる（by_vector に引数を足したとき、在席者相関の
            # テストが別の顔で落ちた）。呼び出し側まで伝播させて即座に表面化させる。
            logger.exception("recall failed (コードの誤り)")
            raise
        except Exception:
            # 運用上の失敗（DB 障害・埋め込み生成の失敗など）はトレース付きで loud に残しつつ
            # degrade する（hot path・会話は落とさない）。keyword_fallback へは流さない
            # （失敗を静かなテキスト検索で masking しない・棚卸し A1）。
            logger.exception("recall failed")
            return []

    async def recall_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall, *a, **kw)

    def recent_feelings(self, n: int = 5) -> list[dict]:
        rows = self._observations._read_observations_by_kind(
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
        rows = self._observations._read_observations_by_kind(
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
        rows = self._observations._read_observations_by_kind(
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
        rows = self._observations._read_observations_by_situated(
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

    # ── これから作り替えるもの（store/ へ移していない） ──────────────────
    # episodes／memory_salience／unfinished_business を触る以下の一群は、
    # Phase 5 で作り替えが決まっているため、いま層へ移していない。
    #   - W（作業記憶）は O からの派生ビューで毎ターン作り直す（[D-記憶単一化]）
    #     ので、memory_salience に溜める形自体が変わる
    #   - 明示リンクとエピソードは WR 拡散想起へ置き換わる（[D-WR拡散想起]）
    # いま移しても Phase 5 で捨てることになる。撤去が確定していないので
    # legacy/ にも入れない。作り替えの形が決まった段で行き先を決める。

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
                    cur.execute("DELETE FROM memory_salience WHERE source='working_memory'")
                for item in recalled:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO memory_salience (id,memory_id,salience,source,context,episode_id,activated_at) "
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
                        "SELECT ms.memory_id,ms.salience,ms.context,ms.episode_id,ms.activated_at,"
                        "o.content FROM memory_salience ms "
                        "JOIN observations o ON o.id=ms.memory_id "
                        "WHERE ms.source='working_memory' ORDER BY ms.salience DESC",
                    )
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("get_working_memory failed: %s", e); return []

    def open_unfinished_business(
        self,
        summary: str,
        source: str = "agent",
        metadata: dict | None = None,
        related_memory_id: str | None = None,
    ) -> str:
        """Create an open unfinished-business record and return its ID."""
        item_id = str(uuid.uuid4())
        now = clock.now_utc_iso()
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
        dynamism = min(0.6, max(confidences))

        has_today = any(m.get("is_today") for m in memories)
        novelty = 0.4 if has_today else 0.2

        top = max(memories, key=lambda m: float(m.get("confidence", 0.0)))
        summary = top["summary"]

        return Coalition(
            source="memory",
            summary=summary,
            dynamism=dynamism,
            urgency=0.1,
            novelty=novelty,
            context_block="[Memory recall]\n" + context,
        )

    # ── Format helpers (unchanged from original) ───────────────────────────

    def format_for_context(self, memories: list[dict]) -> str:
        if not memories: return ""
        lines = ["[過去の記憶（証拠つき）: conf<0.55 は不確か]:"]
        for m in memories:
            fit_s = f" (適合度:{m['fit']:.2f})" if "fit" in m else ""
            conf = float(m.get("confidence", 0.0))
            conf_s = f" conf:{conf:.2f}"
            low = " low-confidence" if conf < 0.55 else ""
            emo  = f" [{m['emotion']}]" if m.get("emotion") and m["emotion"] != "neutral" else ""
            # 12桁（ハイフンを除いた16進）。8桁だと記録が10万件規模でほぼ確実に衝突する。
            # 照合は呼び出し側が対応表で行うので、写し間違いは一致せず件数のずれに出る。
            sid  = str(m.get("memory_id","")).replace("-", "")[:12] or "?"
            lines.append(
                f"- {m.get('date','?')} {m.get('time','?')} id:{sid}{fit_s}{conf_s}{low}"
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
    def _write_store(self) -> ObservationMemory:
        # 話者未解決（声/カメラ/明示のどれも無い＝単独テキスト等）でも書けるよう、
        # 既定話者 DEFAULT_PERSON_ID の記憶へフォールバックする（優先度の floor）。
        # 自動ターン永続化の _active_memory と同じく、書き込みは常に person へ帰属させる。
        return self._manager.get_speaker_memory() or self._manager.get_memory_for(DEFAULT_PERSON_ID)

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

    async def call(self, tool_name: str, tool_input: dict, *,
                   exclude_ids: list[str] | None = None) -> tuple[str, str | None]:
        """`exclude_ids` は内部呼び出し用。自分が出した検索が自分自身を拾うのを防ぐ。"""
        if tool_name == "remember":
            return await self._remember(tool_input)
        if tool_name == "recall":
            return await self._recall(tool_input, exclude_ids=exclude_ids)
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
        # 話者未解決なら既定話者 DEFAULT_PERSON_ID を writer/subject に使う（floor）。
        speaker_id  = self._manager.current_speaker_id or DEFAULT_PERSON_ID
        results: list[str] = []

        # speaker
        if scope in ("speaker", "all"):
            store = self._write_store  # フォールバックで None にならない
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

        # floor：どの scope でも1件も書けなかったら話者／DEFAULT の本命へ落とす
        # （witnessed で在席他者ゼロ等・記憶を落とさない・Slice A の一般化）。
        if not results:
            store = self._write_store
            mem_id, ok = await store.save_async_with_id(
                content, kind="utterance", emotion=emotion,
                image_path=image_path,
                writer_id=speaker_id, subject_id=speaker_id,
                participants=present_ids, scope="speaker",
            )
            if ok:
                results.append(f"[{self._manager.get_person_name(speaker_id)}] 話者（在席者なし）")
                if link_to and mem_id:
                    await store.link_memories_async(mem_id, link_to, link_type=link_type)

        summary = " / ".join(results) if results else "書き込みなし"
        return f"記憶しました: {summary}", None

    async def _recall(self, inp: dict, *,
                      exclude_ids: list[str] | None = None) -> tuple[str, None]:
        query = inp["query"]
        n     = int(inp.get("n", MemoryConfig().recall_k))
        all_results: list[dict] = []

        # agent self
        agent_mem = self._agent_store
        for m in await agent_mem.recall_async(
            query, n=n, exclude_ids=exclude_ids
        ):
            m["_from"] = "自分"
            all_results.append(m)

        # all present persons
        for pid, mem in self._manager.get_all_present_memories():
            name = self._manager.get_person_name(pid)
            for m in await mem.recall_async(
                query, n=n, exclude_ids=exclude_ids
            ):
                m["_from"] = name
                all_results.append(m)

        all_results.sort(key=lambda x: x.get("fit", 0.0), reverse=True)
        top = all_results[:n]

        if not top:
            return "関連する記憶はありません。", None

        lines = []
        for m in top:
            s = f" ({m['fit']:.2f})" if "fit" in m else ""
            lines.append(
                f"[{m.get('_from','?')}] {m['date']} {m['time']}{s} [{m['emotion']}]: {m['summary'][:130]}"
            )
        return "\n".join(lines), None
