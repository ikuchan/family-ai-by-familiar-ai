"""Observation and emotional memory — PostgreSQL + pgvector + person perspectives.

Key changes from original
--------------------------
* Backend : SQLite → PostgreSQL (psycopg2, RealDictCursor)
* Vectors : obs_embeddings keeps BYTEA; situated_embeddings uses vector(384)
* Persons : every row carries person_id; AGENT_SELF_ID for agent-own memories
* Recall  : uses situated_embeddings for person-scoped cosine search via SQL
* Write   : pre-computes situated vectors for all registered persons at save time
* Scope   : remember() accepts scope= "speaker"|"witnessed"|"scene"|"all"
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..db import Database, get_db, vec_to_sql, sql_to_vec
from ..db_migrations import apply_migrations, default_migration_dir
from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID, ALPHA

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager

logger = logging.getLogger(__name__)

DB_PATH_UNUSED = ""          # kept for API compatibility, ignored
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM   = 384
_THUMB_SIZE     = (320, 240)


# ── Helpers ────────────────────────────────────────────────────────────────

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
        prefixed = [f"passage: {t}" for t in texts]
        miss_idx, miss_texts, results = self._lookup(self._d_cache, prefixed)
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
        prefixed = [f"query: {t}" for t in texts]
        miss_idx, miss_texts, results = self._lookup(self._q_cache, prefixed)
        if miss_texts:
            try:
                new = self._model.encode(miss_texts, normalize_embeddings=True,
                                         show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning("encode_query failed: %s", e); return self._zeros(len(texts))
            self._store(self._q_cache, miss_texts, new)
            for j, i in enumerate(miss_idx): results[i] = new[j]
        return results  # type: ignore


# ── ObservationMemory ──────────────────────────────────────────────────────

class ObservationMemory:
    """PostgreSQL-backed memory store scoped to one person_id."""

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
        conn = self._db.conn()
        apply_migrations(conn, default_migration_dir())
        return conn

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

    def _get_perspective_vec(self, person_id: str) -> np.ndarray:
        """Load person's perspective vector from DB. Returns zeros if none."""
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT perspective_vec FROM persons WHERE id = %s", (person_id,))
                row = cur.fetchone()
        if row and row["perspective_vec"]:
            return _decode_vector(bytes(row["perspective_vec"]))
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def _update_perspective_vec(self, person_id: str, mem_vec: np.ndarray, lr: float = 0.05) -> None:
        """Moving-average update of person's perspective vector."""
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
    ) -> None:
        """Compute and store situated vector for one person."""
        p_vec = self._get_perspective_vec(person_id)
        situated = _normalise(mem_vec + ALPHA * p_vec)
        vec_str = vec_to_sql(situated.tolist())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) "
                "VALUES (%s, %s, %s, %s::vector) "
                "ON CONFLICT (obs_id, person_id) DO UPDATE SET vector = EXCLUDED.vector",
                (str(uuid.uuid4()), obs_id, person_id, vec_str),
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
                    upd = cur.execute(
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
            save_date = str(override_date); save_time = "23:59"
            save_ts   = f"{save_date}T23:59:59"
        else:
            save_date = now.strftime("%Y-%m-%d"); save_time = now.strftime("%H:%M")
            save_ts   = now.isoformat()

        participants_json = json.dumps(participants or [], ensure_ascii=False)

        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM observations WHERE id=%s", (event_id,))
                if cur.fetchone():
                    return True
                cur.execute(
                    "INSERT INTO observations "
                    "(id,content,timestamp,date,time,direction,kind,emotion,"
                    " image_path,image_data,person_id,writer_id,subject_id,"
                    " participants_json,scope) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (event_id, content, save_ts, save_date, save_time,
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
            conn.commit()

        # Update this person's perspective vector in background
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._update_perspective_vec(self._person_id, np.array(vec, dtype=np.float32)),
        )
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
            event_id, created_new = self.append_memory_event(
                "memory.save", payload, dedupe_key=dedupe_key,
                queue_job=True, job_type="materialize_observation",
            )
            if dedupe_key and event_id and not created_new:
                return True
            if not materialize_now and event_id:
                return True
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

    def recall(self, query: str, n: int = 3, kind: str | None = None) -> list[dict]:
        """Recall using situated vectors (pgvector cosine search)."""
        try:
            q_vec = np.array(self._embedder.encode_query([query])[0], dtype=np.float32)
            # Apply person's perspective bias to query too
            p_vec = self._get_perspective_vec(self._person_id)
            situated_q = _normalise(q_vec + ALPHA * p_vec)
            q_sql = vec_to_sql(situated_q.tolist())

            kind_clause = "AND o.kind = %s" if kind else ""
            params: list = [q_sql, self._person_id]
            if kind:
                params.append(kind)
            params += [q_sql, n]

            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT o.id, o.content, o.timestamp, o.date, o.time,
                               o.direction, o.kind, o.emotion, o.image_path,
                               COALESCE(o.importance, 1.0) AS importance,
                               1 - (s.vector <=> %s::vector) AS score
                        FROM situated_embeddings s
                        JOIN observations o ON o.id = s.obs_id
                        WHERE s.person_id = %s
                          AND o.superseded_by IS NULL
                          {kind_clause}
                        ORDER BY s.vector <=> %s::vector
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = cur.fetchall()

            if rows:
                return [
                    {
                        "memory_id":        row["id"],
                        "timestamp":        row["timestamp"],
                        "summary":          row["content"],
                        "date":             row["date"],
                        "time":             row["time"],
                        "direction":        row["direction"],
                        "kind":             row["kind"],
                        "source_kind":      row["kind"],
                        "emotion":          row["emotion"],
                        "image_path":       row["image_path"],
                        "score":            float(row["score"]),
                        "confidence":       max(0.0, min(1.0, (float(row["score"]) + 1.0) / 2.0)),
                        "retrieval_method": "situated_cosine",
                    }
                    for row in rows
                ]

            # Fallback: plain keyword search when no situated vectors exist yet
            return self._recall_keyword_fallback(query, n, kind)
        except Exception as e:
            logger.warning("recall failed: %s", e)
            return []

    def _recall_keyword_fallback(self, query: str, n: int, kind: str | None) -> list[dict]:
        keywords = [w for w in query.split() if len(w) > 1][:4]
        if not keywords:
            return []
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
                    SELECT o.id, o.content, o.timestamp, o.date, o.time,
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
                "summary": r["content"], "date": r["date"], "time": r["time"],
                "direction": r["direction"], "kind": r["kind"],
                "source_kind": r["kind"], "emotion": r["emotion"],
                "image_path": r["image_path"],
                "confidence": 0.45, "retrieval_method": "keyword",
            }
            for r in rows
        ]

    async def recall_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall, *a, **kw)

    def recent_feelings(self, n: int = 5) -> list[dict]:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content, date, time, emotion FROM observations "
                        "WHERE kind IN ('feeling','conversation') AND person_id=%s "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (self._person_id, n),
                    )
                    return [{"summary":r["content"],"date":r["date"],"time":r["time"],"emotion":r["emotion"]}
                            for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recent_feelings failed: %s", e); return []

    async def recent_feelings_async(self, n: int = 5):
        return await asyncio.to_thread(self.recent_feelings, n)

    def recall_self_model(self, n: int = 5) -> list[dict]:
        """Always uses AGENT_SELF_ID scope — agent's own self-understanding."""
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content, date, time, emotion FROM observations "
                        "WHERE kind='self_model' AND person_id=%s "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (AGENT_SELF_ID, n),
                    )
                    return [{"summary":r["content"],"date":r["date"],"time":r["time"],"emotion":r["emotion"]}
                            for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recall_self_model failed: %s", e); return []

    async def recall_self_model_async(self, n: int = 5):
        return await asyncio.to_thread(self.recall_self_model, n)

    def recall_curiosities(self, n: int = 5) -> list[dict]:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content, date, time FROM observations "
                        "WHERE kind='curiosity' AND person_id=%s "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (AGENT_SELF_ID, n),
                    )
                    return [{"summary":r["content"],"date":r["date"],"time":r["time"]}
                            for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recall_curiosities failed: %s", e); return []

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
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content,date,time,emotion FROM observations "
                        "WHERE kind='day_summary' AND person_id=%s "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (self._person_id, n),
                    )
                    return [{"summary":r["content"],"date":r["date"],"time":r["time"],"emotion":r["emotion"]}
                            for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recall_day_summaries failed: %s", e); return []

    async def recall_day_summaries_async(self, n=5):
        return await asyncio.to_thread(self.recall_day_summaries, n)

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
                    "WHERE date < %s AND person_id = %s AND superseded_by IS NULL",
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

    def get_linked_memories(self, memory_id: str, direction: str = "both") -> list[dict]:
        try:
            results = []
            with self._db_lock:
                conn = self._ensure_connected()
                if direction in ("out", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.date,o.time,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.target_id "
                            "WHERE ml.source_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend({**dict(r), "link_direction":"→"} for r in cur.fetchall())
                if direction in ("in", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.date,o.time,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.source_id "
                            "WHERE ml.target_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend({**dict(r), "link_direction":"←"} for r in cur.fetchall())
            return results
        except Exception as e:
            logger.warning("get_linked_memories failed: %s", e); return []

    async def get_linked_memories_async(self, *a, **kw):
        return await asyncio.to_thread(self.get_linked_memories, *a, **kw)

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
                f"- {m['date']} {m['time']} id:{sid}{score_s}{conf_s}{low}"
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
        ]

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, str | None]:
        if tool_name == "remember":
            return await self._remember(tool_input)
        if tool_name == "recall":
            return await self._recall(tool_input)
        return f"Unknown memory tool: {tool_name}", None

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
        for m in await agent_mem.recall_async(query, n=n):
            m["_from"] = "自分"
            all_results.append(m)

        # all present persons
        for pid, mem in self._manager.get_all_present_memories():
            name = self._manager.get_person_name(pid)
            for m in await mem.recall_async(query, n=n):
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
