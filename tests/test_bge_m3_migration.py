"""Tests for bge-m3 migration (embedding model swap 384 → 1024)."""

from __future__ import annotations

import os

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

_DB_URL = os.environ["DATABASE_URL"]


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn



def test_embedding_model_is_bge_m3() -> None:
    assert EMBEDDING_MODEL == "BAAI/bge-m3"


def test_embedding_dim_is_1024() -> None:
    assert EMBEDDING_DIM == 1024


# ── Unit: no prefix for bge-m3 ───────────────────────────────────────────────


def test_encode_document_sends_raw_text() -> None:
    """encode_document must pass text as-is (no 'passage:' prefix) to the model."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 1024), dtype=np.float32)

    # 読み込みの状態はモデル資源（MR）の型枠が持つ（出-c）。素の器を作って
    # 読み込み済みに見せる代わりに、普通に組み立ててモデルだけ差し込む。
    em = _EmbeddingModel(model_name=EMBEDDING_MODEL)
    em._mr_model = fake_model

    em.encode_document(["hello world"])

    called_texts = fake_model.encode.call_args[0][0]
    assert called_texts == ["hello world"], (
        f"Expected raw text; got {called_texts!r}"
    )


def test_encode_query_sends_raw_text() -> None:
    """encode_query must pass text as-is (no 'query:' prefix) to the model."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 1024), dtype=np.float32)


    # 読み込みの状態はモデル資源（MR）の型枠が持つ（出-c）。素の器を作って
    # 読み込み済みに見せる代わりに、普通に組み立ててモデルだけ差し込む。
    em = _EmbeddingModel(model_name=EMBEDDING_MODEL)
    em._mr_model = fake_model

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


def test_the_vector_column_is_1024() -> None:
    """`situated_memories.vector` は vector(1024)（020 が確立した姿）。"""
    conn = _pg_conn()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'situated_memories'::regclass
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


def test_the_hnsw_index_exists() -> None:
    """vector に HNSW 索引が張られている（020 が確立した姿・改名は 046）。"""
    conn = _pg_conn()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'situated_memories'
              AND indexname = 'idx_situated_hnsw'
        """)
        row = cur.fetchone()
    conn.close()

    assert row is not None, "idx_situated_hnsw not found after migration"
    assert "hnsw" in row["indexdef"].lower()

# ── 流し直しをやめた理由（044・2026-09-01）─────────────────────────────
# マイグレーションは一度しか流れない。後の版がスキーマを変えれば、前の版はもう流せない。
# 044 が `situated_embeddings` を `situated_memories` へ改名し、`observations` から
# `last_recalled_at` と `groundedness_n` を落としたので、旧名を前提としたマイグレーションを
# テストから呼び出せなくなった。過去のマイグレーションは適用済みの歴史なので書き換えない。
# よって「流して効果を見る」形をやめ、「**いまの姿**を確かめる」形へ寄せた。
# 流したときの一回きりの効果（表が空になる・重複が消える）は事後に確かめられないので、
# そのテストは仕様ごと落とした。
