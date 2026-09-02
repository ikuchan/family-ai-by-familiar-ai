"""Tests for 埋め込み平均ベクトル mu の器と初回推定（平均中心化 C1・未接続）。

計測台帳 §1：コサインを取る前に全埋め込みから共通成分（平均ベクトル mu）を引いて
L2 正規化する。C1 では mu を保存する器（scope 付き複数行）と初回推定だけを作り、
中心化の適用（situated 書き込みと recall クエリ・既存 backfill）は C2。
"""

from __future__ import annotations

import os

import importlib.util
import uuid
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import load_embedding_mean


_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-07-18-026_embedding_means.py"

_DIM = 8  # テスト用の小さな次元（実運用は 1024）


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _run_migration(conn) -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def _clear() -> None:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM obs_embeddings")
        cur.execute("DROP TABLE IF EXISTS embedding_means")
    conn.close()


def _insert_obs_with_vec(cur, vec: np.ndarray) -> None:
    obs_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
        "VALUES (%s,%s,NOW(),%s,%s,%s)",
        (obs_id, "mu test", "unknown", "conversation", "neutral"),
    )
    cur.execute(
        "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s)",
        (obs_id, np.asarray(vec, dtype=np.float32).tobytes()),
    )


# ── 1. テーブルが作られる ───────────────────────────────────────────────────
def test_migration_creates_embedding_means_table() -> None:
    _clear()
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'embedding_means'
        """)
        cols = {r["column_name"] for r in cur.fetchall()}
    conn.close()
    assert {"id", "scope", "scope_key", "dim", "vector", "sample_count", "updated_at"} <= cols


# ── 2. mu が平均として保存される ───────────────────────────────────────────
def test_migration_stores_global_mean() -> None:
    _clear()
    vecs = [
        np.arange(_DIM, dtype=np.float32),
        np.ones(_DIM, dtype=np.float32),
        np.zeros(_DIM, dtype=np.float32),
    ]
    conn = _pg_conn()
    with conn.cursor() as cur:
        for v in vecs:
            _insert_obs_with_vec(cur, v)
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT dim, vector, sample_count FROM embedding_means "
            "WHERE scope='global' AND scope_key=''"
        )
        row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row["dim"] == _DIM
    assert row["sample_count"] == len(vecs)
    stored = np.frombuffer(bytes(row["vector"]), dtype=np.float32)
    expected = np.mean(np.stack(vecs), axis=0)
    assert np.allclose(stored, expected)


# ── 3. 観測0件なら行を作らない ─────────────────────────────────────────────
def test_migration_no_row_when_corpus_empty() -> None:
    _clear()
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM embedding_means")
        n = cur.fetchone()["n"]
    conn.close()
    assert n == 0


# ── 4. 読み出し（次元一致で返す・不一致や行なしは None） ────────────────────
def test_load_embedding_mean_reads_and_guards_dim() -> None:
    _clear()
    conn = _pg_conn()
    with conn.cursor() as cur:
        _insert_obs_with_vec(cur, np.ones(_DIM, dtype=np.float32))
    _run_migration(conn)
    conn.close()

    got = load_embedding_mean(_DIM)
    assert got is not None
    assert np.allclose(got, np.ones(_DIM, dtype=np.float32))
    assert load_embedding_mean(_DIM + 1) is None  # 次元不一致は使わない


def test_load_embedding_mean_none_when_absent() -> None:
    _clear()
    conn = _pg_conn()
    _run_migration(conn)
    conn.close()
    assert load_embedding_mean(_DIM) is None
