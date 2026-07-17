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
import hashlib
import json
import logging
import math
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import psycopg2.extras

from ..config import MemoryConfig
from ..db import Database, get_db, vec_to_sql
from ..mood_register import MoodPAD
from ..db_migrations import apply_migrations, default_migration_dir
from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID, ALPHA
from ..time_decay import DecayState

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager

logger = logging.getLogger(__name__)

DB_PATH_UNUSED = ""          # kept for API compatibility, ignored
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM   = 1024
_THUMB_SIZE     = (320, 240)

# Time-window dedup: identical (person_id, content, kind) within this many seconds
# is treated as a duplicate and silently skipped. Set to 0 to disable.
_CONTENT_DEDUP_WINDOW_SECS: int = int(os.environ.get("MEMORY_DEDUP_WINDOW_SECS", "30"))


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


def _compute_final_score(
    cosine: float,
    ts,
    last_recalled_at,
    recall_count: int,
    activation_a0: float,
    activation_n: int,
    *,
    half_life_days: float,
    floor: float,
) -> float:
    """Compute time-decayed final score via DecayState.

    final_score = cosine × time_score × activation（導出）
    a 軸＝`_derive_activation(a0, n)`（イベント駆動・時間では減らさない）。
    時間減衰は time_score（t 軸）に一元化し、importance の日次減衰は使わない
    （Phase 2 P-1・[D-想起合成]）。
    强化A: recall_count → effective half-life doubles per increment.
    強化B: last_recalled_at as origin_epoch (freshness reset).
    Settings injected from MemoryConfig (time_decay refactor Issue).
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
    return cosine * state.score(now_epoch) * _derive_activation(
        float(activation_a0), int(activation_n)
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _ts_to_date(ts) -> str:
    """Convert a TIMESTAMPTZ row value to YYYY-MM-DD string."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts[:10]
    return ts.date().isoformat()


def _ts_to_time(ts) -> str:
    """Convert a TIMESTAMPTZ row value to HH:MM string."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts[11:16] if len(ts) >= 16 else ts
    return ts.strftime("%H:%M")


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




class _RealDictConnWrapper:
    """Wraps a psycopg2 connection so that cursor() always uses RealDictCursor."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def cursor(self, **kwargs):
        kwargs.setdefault("cursor_factory", psycopg2.extras.RealDictCursor)
        return self._conn.cursor(**kwargs)

    def commit(self):   return self._conn.commit()
    def rollback(self): return self._conn.rollback()
    def close(self):    return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _SQLiteCursorWrapper:
    """Wraps sqlite3.Cursor: adds context-manager support and %s→? translation."""

    def __init__(self, cur) -> None:
        self._cur = cur

    def execute(self, sql: str, params=None) -> None:
        sql = sql.replace("%s", "?")
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._cur.close()

    def __iter__(self):
        return iter(self._cur)


class _SQLiteConnWrapper:
    """Wraps a raw sqlite3.Connection for use in methods that expect psycopg2 style."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def cursor(self, **kwargs) -> "_SQLiteCursorWrapper":
        return _SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):   return self._conn.commit()
    def rollback(self): return self._conn.rollback()
    def close(self):    return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return c @ q


def _encode_vector(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _coerce_to_embedding_dim(vec: np.ndarray) -> np.ndarray:
    """Ensure vec has exactly EMBEDDING_DIM dimensions (pad with zeros or truncate).

    In production embeddings are always EMBEDDING_DIM-dimensional, so this is a
    no-op. In tests that mock the encoder with a small vector the padding keeps
    situated-embedding storage and retrieval consistent.
    """
    if vec.shape[0] == EMBEDDING_DIM:
        return vec
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    n = min(vec.shape[0], EMBEDDING_DIM)
    out[:n] = vec[:n]
    return out


# ── Lazy embedding model ───────────────────────────────────────────────────

class _EmbeddingModel:
    _CACHE_SIZE = 128

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._q_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._d_cache: OrderedDict[str, list[float]] = OrderedDict()

    def pre_warm(self) -> None:
        t = threading.Thread(target=self._load, daemon=True, name="embedding-prewarm")
        t.start()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def _load(self) -> None:
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is None and not self._failed:
                for name in ("sentence_transformers", "huggingface_hub", "transformers"):
                    logging.getLogger(name).setLevel(logging.ERROR)
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
                    logger.info("Embedding model loaded.")
                except Exception as e:
                    self._failed = True
                    logger.warning("Embedding model load failed: %s", e)
                self._ready.set()

    def _zeros(self, n: int) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIM for _ in range(n)]

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _lookup(self, cache: OrderedDict, texts: list[str]):
        results, miss_idx, miss_texts = [], [], []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in cache:
                cache.move_to_end(k)
                results.append(cache[k])
            else:
                miss_idx.append(i); miss_texts.append(t); results.append(None)
        return miss_idx, miss_texts, results

    def _store(self, cache: OrderedDict, texts: list[str], vecs: list[list[float]]) -> None:
        for t, v in zip(texts, vecs):
            k = self._key(t); cache[k] = v; cache.move_to_end(k)
        while len(cache) > self._CACHE_SIZE:
            cache.popitem(last=False)

    def encode_document(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._d_cache, texts)
        if miss_texts:
            try:
                new = self._model.encode(miss_texts, normalize_embeddings=True,
                                         show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning("encode_document failed: %s", e); return self._zeros(len(texts))
            self._store(self._d_cache, miss_texts, new)
            for j, i in enumerate(miss_idx): results[i] = new[j]
        return results  # type: ignore

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._q_cache, texts)
        if miss_texts:
            try:
                new = self._model.encode(miss_texts, normalize_embeddings=True,
                                         show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning("encode_query failed: %s", e); return self._zeros(len(texts))
            self._store(self._q_cache, miss_texts, new)
            for j, i in enumerate(miss_idx): results[i] = new[j]
        return results  # type: ignore


# ── MentalItem ──────────────────────────────────────────────────────────────

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
    """感情一致 e を導出する（課題5 v0.23・ガウシアン・未接続）。

    各 PAD 軸を ε で両端へ寄せロジットで元空間へ戻し、軸重み λ_i つきの
    二乗距離 D²=Σ λ_i (logit(x_obs)-logit(x_mood))² を作り、
    e=exp(-D²/(2σ²)) を返す。完全一致で e=1、遠いほど 0 へ。
    σ・λ_i・ε は設定値（課題5・Config から差し替え可）。

    この段（Phase 2 P-3・スライス1）では `_compute_final_score` へは繋がず、
    e 軸のスコア接続は後続スライスに置く。
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
        return datetime.now().isoformat()

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

    def _get_perspective_vec_with_conn(self, person_id: str, conn) -> np.ndarray:
        """Load perspective vector using an already-open connection (no lock)."""
        with conn.cursor() as cur:
            cur.execute("SELECT perspective_vec FROM persons WHERE id = %s", (person_id,))
            row = cur.fetchone()
        if row and row["perspective_vec"]:
            return _coerce_to_embedding_dim(_decode_vector(bytes(row["perspective_vec"])))
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def _get_perspective_vec(self, person_id: str) -> np.ndarray:
        """Load person's perspective vector from DB. Returns zeros if none."""
        with self._db_lock:
            conn = self._ensure_connected()
            return self._get_perspective_vec_with_conn(person_id, conn)

    def _update_perspective_vec(self, person_id: str, mem_vec: np.ndarray, lr: float = 0.05) -> None:
        """Moving-average update of person's perspective vector."""
        mem_vec = _coerce_to_embedding_dim(mem_vec)
        old = self._get_perspective_vec(person_id)
        new = _normalise((1.0 - lr) * old + lr * mem_vec)
        blob = _encode_vector(new.tolist())
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE persons SET perspective_vec = %s, updated_at = %s WHERE id = %s",
                    (blob, self._now(), person_id),
                )
            conn.commit()

    def _upsert_situated_embedding(
        self,
        conn,
        obs_id: str,
        person_id: str,
        mem_vec: np.ndarray,
        relation_key: str = "presence",
    ) -> None:
        """Compute and store situated vector for one person under a relation_key.

        relation_key は関係の帳簿ラベル（[D-在席相関/V2]）。同定キーは
        (obs_id, person_id, relation_key) で、同じ関係の再計算は vector を更新する。
        生成の多型化（speaker/subject）は後続スライス。
        """
        mem_vec = _coerce_to_embedding_dim(mem_vec)
        p_vec = self._get_perspective_vec_with_conn(person_id, conn)
        situated = _normalise(mem_vec + ALPHA * p_vec)
        vec_str = vec_to_sql(situated.tolist())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_embeddings (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s, %s, %s, %s::vector, %s) "
                "ON CONFLICT (obs_id, person_id, relation_key) DO UPDATE SET vector = EXCLUDED.vector",
                (str(uuid.uuid4()), obs_id, person_id, vec_str, relation_key),
            )

    def _refresh_situated_embeddings(self, conn, obs_id: str, mem_vec: np.ndarray) -> None:
        """Pre-compute situated vectors for ALL registered persons + agent self."""
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM persons")
            person_ids = [row["id"] for row in cur.fetchall()]
        for pid in person_ids:
            self._upsert_situated_embedding(conn, obs_id, pid, mem_vec)
        # Always include AGENT_SELF_ID
        if AGENT_SELF_ID not in person_ids:
            self._upsert_situated_embedding(conn, obs_id, AGENT_SELF_ID, mem_vec)

    # ── Event / job queue ──────────────────────────────────────────────────

    def _enqueue_job(self, conn, event_id: str, job_type: str, now: str) -> bool:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_jobs "
                    "(job_id,event_id,job_type,status,attempts,available_at,last_error,created_at,updated_at) "
                    "VALUES (%s,%s,%s,'pending',0,%s,NULL,%s,%s)",
                    (str(uuid.uuid4()), event_id, job_type, now, now, now),
                )
            return True
        except Exception:
            return False

    def append_memory_event(
        self,
        event_type: str,
        payload: dict,
        dedupe_key: str | None = None,
        queue_job: bool = True,
        job_type: str = "materialize_observation",
    ) -> tuple[str | None, bool]:
        now = self._now()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",",":"), sort_keys=True)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                if dedupe_key:
                    with conn.cursor() as cur:
                        cur.execute("SELECT event_id FROM memory_events WHERE dedupe_key = %s", (dedupe_key,))
                        row = cur.fetchone()
                    if row:
                        eid = str(row["event_id"])
                        if queue_job:
                            self._enqueue_job(conn, eid, job_type, now)
                        conn.commit()
                        return eid, False
                eid = str(uuid.uuid4())
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_events (event_id,created_at,event_type,dedupe_key,payload_json,person_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s)",
                        (eid, now, event_type, dedupe_key, payload_json, self._person_id),
                    )
                if queue_job:
                    self._enqueue_job(conn, eid, job_type, now)
                conn.commit()
                return eid, True
        except Exception as e:
            logger.warning("append_memory_event failed: %s", e)
            return None, False

    async def append_memory_event_async(self, *a, **kw):
        return await asyncio.to_thread(self.append_memory_event, *a, **kw)

    def claim_pending_jobs(self, limit: int = 10) -> list[dict]:
        now = self._now()
        claimed = []
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT j.job_id,j.event_id,j.job_type,j.attempts, "
                    "e.event_type,e.payload_json "
                    "FROM memory_jobs j JOIN memory_events e ON e.event_id = j.event_id "
                    "WHERE j.status='pending' AND j.available_at <= %s "
                    "ORDER BY j.created_at LIMIT %s",
                    (now, limit),
                )
                rows = cur.fetchall()
            for row in rows:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memory_jobs SET status='running',attempts=attempts+1,updated_at=%s "
                        "WHERE job_id=%s AND status='pending' RETURNING job_id",
                        (now, row["job_id"]),
                    )
                    if cur.rowcount != 1:
                        continue
                try:
                    payload = json.loads(row["payload_json"])
                except Exception:
                    payload = {"raw_payload": row["payload_json"]}
                claimed.append({
                    "job_id":     row["job_id"],
                    "event_id":   row["event_id"],
                    "job_type":   row["job_type"],
                    "attempts":   int(row["attempts"]) + 1,
                    "event_type": row["event_type"],
                    "payload":    payload,
                })
            conn.commit()
        return claimed

    def mark_job_done(self, job_id: str) -> bool:
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memory_jobs SET status='done',updated_at=%s,last_error=NULL WHERE job_id=%s",
                    (self._now(), job_id),
                )
            conn.commit()
            return True

    def mark_job_failed(self, job_id: str, error: str, retry_delay: float = 10.0, max_attempts: int = 3) -> str:
        now = datetime.now()
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT attempts FROM memory_jobs WHERE job_id=%s", (job_id,))
                row = cur.fetchone()
            if row is None:
                return "missing"
            attempts = int(row["attempts"])
            status = "dead_letter" if attempts >= max_attempts else "pending"
            avail = now.isoformat() if status == "dead_letter" else \
                    (now + timedelta(seconds=max(retry_delay, 0.0))).isoformat()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memory_jobs SET status=%s,available_at=%s,last_error=%s,updated_at=%s WHERE job_id=%s",
                    (status, avail, error[:500], now.isoformat(), job_id),
                )
            conn.commit()
        return status

    # ── Core save ──────────────────────────────────────────────────────────

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

        if not content:
            return False

        image_data = _encode_image(image_path) if image_path else None
        vec = self._embedder.encode_document([content])[0]
        blob = _encode_vector(vec)
        now = datetime.now()
        if override_date:
            d = datetime.strptime(str(override_date)[:10], "%Y-%m-%d")
            save_ts = d.replace(hour=23, minute=59, second=59)
        else:
            save_ts = now

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
                    " participants_json,scope) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (event_id, content, save_ts,
                     direction, kind, emotion, image_path, image_data,
                     self._person_id,
                     writer_id or self._person_id,
                     subject_id or self._person_id,
                     participants_json, scope),
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
    ) -> bool:
        payload = dict(content=content, direction=direction, kind=kind,
                       emotion=emotion, image_path=image_path,
                       override_date=override_date)
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
        payload = dict(content=content, direction=kwargs.get("direction","unknown"),
                       kind=kwargs.get("kind","observation"), emotion=kwargs.get("emotion","neutral"),
                       image_path=kwargs.get("image_path"), override_date=kwargs.get("override_date"))
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

    def materialize_event(self, event_id: str) -> bool:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT event_type, payload_json FROM memory_events WHERE event_id=%s",
                        (event_id,),
                    )
                    row = cur.fetchone()
            if not row:
                return False
            payload = json.loads(row["payload_json"])
            if row["event_type"] == "memory.save":
                return self._materialize_save_event(event_id, payload)
            return False
        except Exception as e:
            logger.warning("materialize_event failed: %s", e)
            return False

    # ── Recall ─────────────────────────────────────────────────────────────

    def recall(self, query: str, n: int = 3, kind: str | None = None,
               min_score: float = 0.0,
               recall_mode: str = "system") -> list[dict]:
        """Recall using situated vectors (pgvector cosine search).

        min_score:   cosine similarity threshold (Issue A).
        recall_mode: reinforcement mode (Issue B).
          "conversation" → recall_count += 1 AND last_recalled_at = now()
          "spontaneous"  → last_recalled_at = now() only
          "system"       → no reinforcement (default)

        Scoring: final_score = cosine × time_score × importance
        Results are re-sorted by final_score before returning.
        """
        try:
            q_vec = _coerce_to_embedding_dim(
                np.array(self._embedder.encode_query([query])[0], dtype=np.float32)
            )
            p_vec = self._get_perspective_vec(self._person_id)
            situated_q = _normalise(q_vec + ALPHA * p_vec)
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
                for row in rows:
                    cosine = float(row["score"])
                    final = _compute_final_score(
                        cosine,
                        row["timestamp"],
                        row["last_recalled_at"],
                        int(row["recall_count"]),
                        float(row["activation_a0"]),
                        int(row["activation_n"]),
                        half_life_days=_cfg.recall_half_life_days,
                        floor=_cfg.recall_time_floor,
                    )
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
                    })
                results.sort(key=lambda r: r["score"], reverse=True)

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

    def recall_semantic_facts(self, query: str, n: int = 5) -> list[dict]:
        like = f"%{query.strip()}%" if query.strip() else "%"
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT fact_key,fact_text,source_memory_id,confidence,tags,last_seen_at "
                        "FROM semantic_facts "
                        "WHERE person_id=%s AND (%s='%%' OR fact_text LIKE %s OR tags LIKE %s) "
                        "ORDER BY CASE WHEN fact_text LIKE %s THEN 0 ELSE 1 END, last_seen_at DESC "
                        "LIMIT %s",
                        (self._person_id, like, like, like, like, n),
                    )
                    return [
                        {"key":r["fact_key"],"summary":r["fact_text"],
                         "source_memory_id":r["source_memory_id"],
                         "confidence":float(r["confidence"]),"tags":r["tags"],
                         "last_seen_at":r["last_seen_at"]}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_semantic_facts failed: %s", e); return []

    async def recall_semantic_facts_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall_semantic_facts, *a, **kw)

    def _upsert_semantic_fact_locked(
        self,
        conn: "_RealDictConnWrapper",
        key: str,
        text: str,
        confidence: float = 0.5,
        source_memory_id: str | None = None,
        tags: str = "",
    ) -> None:
        """Upsert a semantic fact inside an already-held lock; record a revision if the text changes."""
        now = datetime.now().isoformat()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, fact_text, confidence FROM semantic_facts "
                "WHERE person_id=%s AND fact_key=%s",
                (self._person_id, key),
            )
            existing = cur.fetchone()
        if existing and existing["fact_text"] != text:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_revisions "
                    "(id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                    "VALUES (%s,'semantic_fact',%s,%s,%s,%s,%s,%s,'upsert',%s)",
                    (str(uuid.uuid4()), key,
                     existing["fact_text"], text,
                     float(existing["confidence"]), confidence,
                     source_memory_id, now),
                )
        if existing:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE semantic_facts SET fact_text=%s,confidence=%s,"
                    "source_memory_id=%s,tags=%s,updated_at=%s,last_seen_at=%s "
                    "WHERE person_id=%s AND fact_key=%s",
                    (text, confidence, source_memory_id, tags, now, now,
                     self._person_id, key),
                )
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO semantic_facts "
                    "(id,fact_key,fact_text,source_memory_id,confidence,tags,"
                    "last_seen_at,created_at,updated_at,person_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), key, text, source_memory_id,
                     confidence, tags, now, now, now, self._person_id),
                )

    def _upsert_behavior_policy_locked(
        self,
        conn: "_RealDictConnWrapper",
        key: str,
        text: str,
        trigger_context: str = "",
        action_hint: str = "",
        confidence: float = 0.5,
        source_memory_id: str | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, policy_text, confidence FROM behavior_policies "
                "WHERE policy_key=%s AND person_id=%s",
                (key, self._person_id),
            )
            existing = cur.fetchone()
        if existing and existing["policy_text"] != text:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_revisions "
                    "(id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                    "VALUES (%s,'behavior_policy',%s,%s,%s,%s,%s,%s,'upsert',%s)",
                    (str(uuid.uuid4()), key,
                     existing["policy_text"], text,
                     float(existing["confidence"]), confidence,
                     source_memory_id, now),
                )
        if existing:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE behavior_policies "
                    "SET policy_text=%s,trigger_context=%s,action_hint=%s,"
                    "confidence=%s,source_memory_id=%s,updated_at=%s,last_seen_at=%s "
                    "WHERE policy_key=%s AND person_id=%s",
                    (text, trigger_context, action_hint, confidence, source_memory_id,
                     now, now, key, self._person_id),
                )
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO behavior_policies "
                    "(id,policy_key,policy_text,trigger_context,action_hint,"
                    "source_memory_id,confidence,last_seen_at,created_at,updated_at,person_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), key, text, trigger_context, action_hint,
                     source_memory_id, confidence, now, now, now, self._person_id),
                )

    def _project_observation(
        self, conn: "_RealDictConnWrapper", obs_id: str, content: str, kind: str, emotion: str
    ) -> None:
        try:
            if kind == "self_model":
                self._upsert_semantic_fact_locked(
                    conn, "self_model:core", content,
                    confidence=0.85, source_memory_id=obs_id, tags="self_model",
                )
            elif kind == "curiosity":
                self._upsert_behavior_policy_locked(
                    conn, "curiosity:active", content,
                    trigger_context="idle", action_hint="look_around",
                    confidence=0.75, source_memory_id=obs_id,
                )
            elif kind == "conversation" and emotion == "moved":
                self._upsert_behavior_policy_locked(
                    conn, "conversation:support", content,
                    trigger_context="conversation", action_hint="respond_supportively",
                    confidence=0.80, source_memory_id=obs_id,
                )
        except Exception as e:
            logger.warning("_project_observation failed: %s", e)

    def adjust_behavior_policy_confidence(
        self, key: str, delta: float, reason: str = ""
    ):
        try:
            now = datetime.now().isoformat()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, policy_text, confidence FROM behavior_policies "
                        "WHERE policy_key=%s AND person_id=%s",
                        (key, self._person_id),
                    )
                    row = cur.fetchone()
                if not row:
                    return None
                new_conf = max(0.0, min(1.0, float(row["confidence"]) + delta))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_revisions "
                        "(id,entity_type,entity_key,previous_text,new_text,"
                        "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                        "VALUES (%s,'behavior_policy',%s,%s,%s,%s,%s,NULL,%s,%s)",
                        (str(uuid.uuid4()), key,
                         row["policy_text"], row["policy_text"],
                         float(row["confidence"]), new_conf, reason, now),
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE behavior_policies SET confidence=%s,updated_at=%s "
                        "WHERE policy_key=%s AND person_id=%s",
                        (new_conf, now, key, self._person_id),
                    )
                conn.commit()
            return new_conf
        except Exception as e:
            logger.warning("adjust_behavior_policy_confidence failed: %s", e); return None

    async def adjust_behavior_policy_confidence_async(
        self,
        key: str,
        delta: float,
        reason: str = "",
        policy_text: str = "",
        trigger_context: str = "",
        action_hint: str = "",
    ):
        """Async wrapper: adjust confidence, upserting the policy if policy_text is given."""
        def _run():
            if policy_text:
                try:
                    with self._db_lock:
                        conn = self._ensure_connected()
                        self._upsert_behavior_policy_locked(
                            conn, key, policy_text,
                            trigger_context=trigger_context,
                            action_hint=action_hint,
                        )
                        conn.commit()
                except Exception as e:
                    logger.warning("adjust_behavior_policy_confidence_async upsert failed: %s", e)
            return self.adjust_behavior_policy_confidence(key, delta, reason)
        return await asyncio.to_thread(_run)

    def adjust_semantic_fact_confidence(
        self, key: str, delta: float, reason: str = ""
    ):
        try:
            now = datetime.now().isoformat()
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, fact_text, confidence FROM semantic_facts "
                        "WHERE fact_key=%s AND person_id=%s",
                        (key, self._person_id),
                    )
                    row = cur.fetchone()
                if not row:
                    return None
                new_conf = max(0.0, min(1.0, float(row["confidence"]) + delta))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_revisions "
                        "(id,entity_type,entity_key,previous_text,new_text,"
                        "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                        "VALUES (%s,'semantic_fact',%s,%s,%s,%s,%s,NULL,%s,%s)",
                        (str(uuid.uuid4()), key,
                         row["fact_text"], row["fact_text"],
                         float(row["confidence"]), new_conf, reason, now),
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE semantic_facts SET confidence=%s,updated_at=%s "
                        "WHERE fact_key=%s AND person_id=%s",
                        (new_conf, now, key, self._person_id),
                    )
                conn.commit()
            return new_conf
        except Exception as e:
            logger.warning("adjust_semantic_fact_confidence failed: %s", e); return None

    async def adjust_semantic_fact_confidence_async(self, key: str, delta: float, reason: str = ""):
        return await asyncio.to_thread(self.adjust_semantic_fact_confidence, key, delta, reason)

    def get_dates_with_observations(self, days: int = 7) -> list[str]:
        """Return distinct dates (YYYY-MM-DD) that have observations within the last N days."""
        try:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
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
                    "time": _ts_to_time(row["timestamp"]),
                })
            return result
        except Exception as e:
            logger.warning("get_observations_for_date failed: %s", e); return []

    def recall_revisions(
        self,
        entity_type: str = "semantic_fact",
        entity_key: str | None = None,
        n: int = 50,
    ) -> list[dict]:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                params: list = [entity_type]
                sql = (
                    "SELECT id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at "
                    "FROM memory_revisions WHERE entity_type=%s"
                )
                if entity_key:
                    sql += " AND entity_key=%s"
                    params.append(entity_key)
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(n)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recall_revisions failed: %s", e); return []

    def recall_behavior_policies(self, query: str, n: int = 5) -> list[dict]:
        like = f"%{query.strip()}%" if query.strip() else "%"
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT policy_key,policy_text,trigger_context,action_hint,"
                        "source_memory_id,confidence,last_seen_at "
                        "FROM behavior_policies "
                        "WHERE person_id=%s AND (%s='%%' OR policy_text LIKE %s "
                        "   OR trigger_context LIKE %s OR action_hint LIKE %s) "
                        "ORDER BY CASE WHEN policy_text LIKE %s THEN 0 ELSE 1 END, last_seen_at DESC "
                        "LIMIT %s",
                        (self._person_id, like, like, like, like, like, n),
                    )
                    return [
                        {"key":r["policy_key"],"summary":r["policy_text"],
                         "trigger_context":r["trigger_context"],"action_hint":r["action_hint"],
                         "source_memory_id":r["source_memory_id"],
                         "confidence":float(r["confidence"]),"last_seen_at":r["last_seen_at"]}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_behavior_policies failed: %s", e); return []

    async def recall_behavior_policies_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall_behavior_policies, *a, **kw)

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

    def link_memories(self, src: str, tgt: str, link_type: str = "related", note: str | None = None) -> bool:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_links (id,source_id,target_id,link_type,note,created_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (str(uuid.uuid4()), src, tgt, link_type, note, self._now()),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning("link_memories failed: %s", e); return False

    async def link_memories_async(self, *a, **kw):
        return await asyncio.to_thread(self.link_memories, *a, **kw)

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

    def get_linked_memories(self, memory_id: str, direction: str = "both") -> list[dict]:
        try:
            results = []
            with self._db_lock:
                conn = self._ensure_connected()
                if direction in ("out", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.timestamp,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.target_id "
                            "WHERE ml.source_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend(
                            {**dict(r), "date": _ts_to_date(r["timestamp"]), "time": _ts_to_time(r["timestamp"]), "link_direction": "→"}
                            for r in cur.fetchall()
                        )
                if direction in ("in", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.timestamp,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.source_id "
                            "WHERE ml.target_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend(
                            {**dict(r), "date": _ts_to_date(r["timestamp"]), "time": _ts_to_time(r["timestamp"]), "link_direction": "←"}
                            for r in cur.fetchall()
                        )
            return results
        except Exception as e:
            logger.warning("get_linked_memories failed: %s", e); return []

    async def get_linked_memories_async(self, *a, **kw):
        return await asyncio.to_thread(self.get_linked_memories, *a, **kw)

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
                        {**dict(r), "date": _ts_to_date(r["timestamp"]), "time": _ts_to_time(r["timestamp"])}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_on_this_day failed: %s", e); return []

    async def recall_on_this_day_async(self, month: int, day: int, n: int = 5) -> list[dict]:
        return await asyncio.to_thread(self.recall_on_this_day, month, day, n)

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

    def open_unfinished_business(
        self,
        summary: str,
        source: str = "agent",
        metadata: dict | None = None,
        related_memory_id: str | None = None,
    ) -> str:
        """Create an open unfinished-business record and return its ID."""
        item_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
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

    def format_semantic_facts_for_context(self, facts: list[dict]) -> str:
        if not facts: return ""
        lines = ["[安定した事実（semantic memory）]:"]
        for f in facts:
            lines.append(f"- conf:{float(f.get('confidence',0)):.2f} key:{str(f.get('key','?'))[:24]}: {str(f.get('summary',''))[:140]}")
        return "\n".join(lines)

    def format_behavior_policies_for_context(self, policies: list[dict]) -> str:
        if not policies: return ""
        lines = ["[行動方針（policy memory）]:"]
        for p in policies:
            lines.append(f"- conf:{float(p.get('confidence',0)):.2f} trigger:{str(p.get('trigger_context',''))[:24]} action:{str(p.get('action_hint',''))[:32]}: {str(p.get('summary',''))[:140]}")
        return "\n".join(lines)

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
