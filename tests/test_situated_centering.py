"""Tests for 平均中心化の適用（C2・挙動変化）。

計測台帳 §1：コサインを取る前に mu を引いて L2 正規化する。書き込み（situated 生成）と
問い合わせ（recall のクエリ）の**両方**に同じ変換をかけ、既存 situated も同じ空間へ移す。
mu が無い／次元不一致なら中心化しない（フォールバック）。
"""

from __future__ import annotations

import importlib.util
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import (
    EMBEDDING_DIM,
    ObservationMemory,
    _EmbeddingModel,
    _normalise,
    _situated_vector,
)
from familiar_agent.person_memory_manager import ALPHA


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"
_BACKFILL_MIGRATION = "2026-07-18-027_center_situated_embeddings.py"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _fixed_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=EMBEDDING_DIM).astype(np.float32)


def _set_mu(mu: np.ndarray) -> None:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO embedding_means (scope, scope_key, dim, vector, sample_count, updated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (scope, scope_key) DO UPDATE SET "
            "  dim=EXCLUDED.dim, vector=EXCLUDED.vector, "
            "  sample_count=EXCLUDED.sample_count, updated_at=EXCLUDED.updated_at",
            ("global", "", int(mu.size), np.asarray(mu, dtype=np.float32).tobytes(),
             1, datetime.now(timezone.utc)),
        )
    conn.close()


def _clear_mu() -> None:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM embedding_means WHERE scope='global'")
    conn.close()


def _read_situated(obs_id: str, person_id: str) -> np.ndarray:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT vector FROM situated_embeddings WHERE obs_id=%s AND person_id=%s",
            (obs_id, person_id),
        )
        row = cur.fetchone()
    conn.close()
    assert row is not None, "situated row not found"
    raw = row["vector"]
    if isinstance(raw, str):
        return np.array([float(x) for x in raw.strip("[]").split(",")], dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def _ensure_person(person_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id,name,display_name,created_at,updated_at) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (person_id, f"t-{person_id[:8]}", "Test", now, now),
        )
    conn.close()


def _mem(person_id: str, doc_vec: np.ndarray) -> ObservationMemory:
    _ensure_person(person_id)
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory(person_id=person_id)
    return mem


# ── 1. 純関数：mu あり／なし ───────────────────────────────────────────────
def test_situated_vector_centers_when_mu_present() -> None:
    mem_vec = _fixed_vec(1)
    p_vec = _fixed_vec(2)
    mu = _fixed_vec(3)
    got = _situated_vector(mem_vec, p_vec, mu)
    expected = _normalise(mem_vec + ALPHA * p_vec - mu)
    assert np.allclose(got, expected)


def test_situated_vector_falls_back_when_mu_none() -> None:
    mem_vec = _fixed_vec(1)
    p_vec = _fixed_vec(2)
    got = _situated_vector(mem_vec, p_vec, None)
    expected = _normalise(mem_vec + ALPHA * p_vec)
    assert np.allclose(got, expected)


# ── 2. 書き込みが中心化される ─────────────────────────────────────────────
def test_save_stores_centered_situated() -> None:
    mu = _fixed_vec(10)
    _set_mu(mu)
    doc_vec = _fixed_vec(11)
    person_id = str(uuid.uuid4())
    mem = _mem(person_id, doc_vec)

    with patch.object(_EmbeddingModel, "encode_document", return_value=[doc_vec]):
        obs_id, _ = mem.save_with_id("centering write test " + uuid.uuid4().hex,
                                     materialize_now=True)
    assert obs_id is not None

    stored = _read_situated(obs_id, person_id)
    expected = _normalise(doc_vec - mu)  # perspective は未設定でゼロ
    assert np.allclose(stored, expected, atol=1e-5)


# ── 3. mu が無ければ従来式（フォールバック） ───────────────────────────────
def test_save_without_mu_uses_legacy_formula() -> None:
    _clear_mu()
    doc_vec = _fixed_vec(20)
    person_id = str(uuid.uuid4())
    mem = _mem(person_id, doc_vec)

    with patch.object(_EmbeddingModel, "encode_document", return_value=[doc_vec]):
        obs_id, _ = mem.save_with_id("legacy write test " + uuid.uuid4().hex,
                                     materialize_now=True)
    stored = _read_situated(obs_id, person_id)
    assert np.allclose(stored, _normalise(doc_vec), atol=1e-5)


# ── 4. 書き込みと問い合わせが同じ空間（反証：ずれると自分が見つからない） ────
def test_write_and_query_share_space() -> None:
    mu = _fixed_vec(30)
    _set_mu(mu)
    doc_vec = _fixed_vec(31)
    person_id = str(uuid.uuid4())
    mem = _mem(person_id, doc_vec)
    content = "same space test " + uuid.uuid4().hex

    with patch.object(_EmbeddingModel, "encode_document", return_value=[doc_vec]):
        mem.save_with_id(content, materialize_now=True)

    with patch.object(_EmbeddingModel, "encode_query", return_value=[doc_vec]):
        results = mem.recall(content, n=5)

    assert any(r["summary"] == content for r in results), "自分自身が想起されない＝空間がずれている"


# ── 5. backfill：既存 situated が中心化後へ書き換わる ──────────────────────
def _run_backfill(conn) -> None:
    path = Path(__file__).parent.parent / "migration" / _BACKFILL_MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def test_backfill_centers_existing_situated() -> None:
    mu = _fixed_vec(40)
    doc_vec = _fixed_vec(41)
    person_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())
    legacy = _normalise(doc_vec)

    _ensure_person(person_id)
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s)",
            (obs_id, "backfill target", "unknown", "conversation", "neutral", person_id),
        )
        cur.execute(
            "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s,%s)",
            (obs_id, doc_vec.tobytes()),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), obs_id, person_id,
             "[" + ",".join(str(float(x)) for x in legacy) + "]"),
        )
    conn.close()

    _set_mu(mu)
    conn = _pg_conn()
    _run_backfill(conn)
    conn.close()

    stored = _read_situated(obs_id, person_id)
    assert np.allclose(stored, _normalise(doc_vec - mu), atol=1e-5)
