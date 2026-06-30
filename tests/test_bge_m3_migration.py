"""Tests for bge-m3 migration (embedding model swap 384 → 1024)."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    _EmbeddingModel,
    _coerce_to_embedding_dim,
)

_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _run_migration(conn) -> None:
    migration_path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-06-29-020_bge_m3_situated_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("bge_m3_migration", migration_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


# ── Unit: constants ──────────────────────────────────────────────────────────


def test_embedding_model_is_bge_m3() -> None:
    assert EMBEDDING_MODEL == "BAAI/bge-m3"


def test_embedding_dim_is_1024() -> None:
    assert EMBEDDING_DIM == 1024


# ── Unit: no prefix for bge-m3 ───────────────────────────────────────────────


def test_encode_document_sends_raw_text() -> None:
    """encode_document must pass text as-is (no 'passage:' prefix) to the model."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 1024), dtype=np.float32)

    em = _EmbeddingModel.__new__(_EmbeddingModel)
    em._model = fake_model
    em._failed = False
    em._ready = MagicMock()
    em._d_cache = {}  # type: ignore[assignment]
    from collections import OrderedDict
    em._d_cache = OrderedDict()
    em._q_cache = OrderedDict()
    em._CACHE_SIZE = 512
    em._model_name = EMBEDDING_MODEL
    em._lock = __import__("threading").Lock()

    em.encode_document(["hello world"])

    called_texts = fake_model.encode.call_args[0][0]
    assert called_texts == ["hello world"], (
        f"Expected raw text; got {called_texts!r}"
    )


def test_encode_query_sends_raw_text() -> None:
    """encode_query must pass text as-is (no 'query:' prefix) to the model."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 1024), dtype=np.float32)

    from collections import OrderedDict

    em = _EmbeddingModel.__new__(_EmbeddingModel)
    em._model = fake_model
    em._failed = False
    em._ready = MagicMock()
    em._d_cache = OrderedDict()
    em._q_cache = OrderedDict()
    em._CACHE_SIZE = 512
    em._model_name = EMBEDDING_MODEL
    em._lock = __import__("threading").Lock()

    em.encode_query(["how are you"])

    called_texts = fake_model.encode.call_args[0][0]
    assert called_texts == ["how are you"], (
        f"Expected raw text; got {called_texts!r}"
    )


# ── Unit: _coerce_to_embedding_dim with 1024 ────────────────────────────────


def test_pad_or_truncate_1024_is_noop() -> None:
    vec = np.ones(1024, dtype=np.float32)
    result = _coerce_to_embedding_dim(vec)
    assert result.shape[0] == 1024
    assert result is vec  # same object: no copy needed


def test_pad_or_truncate_pads_short_vector() -> None:
    vec = np.ones(8, dtype=np.float32)
    result = _coerce_to_embedding_dim(vec)
    assert result.shape[0] == 1024
    assert np.all(result[:8] == 1.0)
    assert np.all(result[8:] == 0.0)


def test_pad_or_truncate_truncates_long_vector() -> None:
    vec = np.ones(2048, dtype=np.float32)
    result = _coerce_to_embedding_dim(vec)
    assert result.shape[0] == 1024


# ── DB: migration schema check ───────────────────────────────────────────────


def test_migration_vector_column_becomes_1024() -> None:
    """After the migration, situated_embeddings.vector must be vector(1024)."""
    conn = _pg_conn()
    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'situated_embeddings'::regclass
              AND attname = 'vector'
              AND attnum > 0
        """)
        row = cur.fetchone()
    conn.close()

    assert row is not None, "vector column not found"
    # pgvector encodes dimension as atttypmod; 1024 → atttypmod = 1024
    assert row["atttypmod"] == 1024, (
        f"Expected atttypmod=1024, got {row['atttypmod']}"
    )


def test_migration_hnsw_index_recreated() -> None:
    """After the migration, an HNSW index on vector must exist."""
    conn = _pg_conn()
    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'situated_embeddings'
              AND indexname = 'idx_se_hnsw'
        """)
        row = cur.fetchone()
    conn.close()

    assert row is not None, "idx_se_hnsw not found after migration"
    assert "hnsw" in row["indexdef"].lower()


def test_migration_clears_situated_embeddings() -> None:
    """Migration must leave situated_embeddings empty (re-embedding fills it)."""

    pid = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (pid, f"test_{pid[:8]}", f"label_{pid[:8]}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        cur.execute(
            "INSERT INTO observations "
            "(id,content,timestamp,direction,kind,emotion,person_id,writer_id,subject_id,"
            " participants_json,scope) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, "migration test", "unknown", "utterance", "neutral",
             pid, pid, pid, "[]", "speaker"),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) "
            "VALUES (%s, %s, %s, %s::vector)",
            (str(uuid.uuid4()), obs_id, pid,
             "[" + ",".join(["0"] * 1024) + "]"),
        )
    conn.commit()

    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM situated_embeddings")
        n = cur.fetchone()["n"]
    conn.close()

    assert n == 0, f"Expected empty table after migration, got {n} rows"


def test_migration_clears_obs_embeddings() -> None:
    """Migration must leave obs_embeddings empty (re-embedding fills it)."""
    from familiar_agent.db import vec_to_sql

    pid = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())

    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (pid, f"test_{pid[:8]}", f"label_{pid[:8]}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        cur.execute(
            "INSERT INTO observations "
            "(id,content,timestamp,direction,kind,emotion,person_id,writer_id,subject_id,"
            " participants_json,scope) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, "obs embedding test", "unknown", "utterance", "neutral",
             pid, pid, pid, "[]", "speaker"),
        )
        fake_blob = vec_to_sql(np.zeros(1024).tolist())
        cur.execute(
            "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (obs_id, fake_blob),
        )
    conn.commit()

    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM obs_embeddings")
        n = cur.fetchone()["n"]
    conn.close()

    assert n == 0, f"Expected empty obs_embeddings after migration, got {n} rows"
