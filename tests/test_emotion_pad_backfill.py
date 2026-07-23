"""Tests for 既存行の label→PAD backfill（書き込み PAD 化 W1b・移行専用）。

W1a（024）で追加した PAD 列は既存行が既定0.5のまま。W1b（025）は確定した
12ラベル→4軸の写像で既存行の PAD を埋める（移行専用・実行時 φ ではない）。
表に無いラベルは既定0.5のまま。W2（評価器が PAD 直接出力）は後続。
"""

from __future__ import annotations

import os

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = os.environ["DATABASE_URL"]

_MIGRATION_DIR = Path(__file__).parent.parent / "migration"

_EXPECTED_LABELS = {
    "happy", "excited", "curious", "moved", "surprised", "nostalgic",
    "relieved", "tender", "playful", "proud", "sad", "neutral",
}


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _load_migration(filename: str):
    path = _MIGRATION_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_columns_migration(conn) -> None:
    _load_migration("2026-07-16-024_observation_emotion_pad.py").upgrade(conn)
    conn.commit()


def _run_backfill_migration(conn):
    mod = _load_migration("2026-07-16-025_backfill_emotion_pad.py")
    mod.upgrade(conn)
    conn.commit()
    return mod


def _insert(cur, obs_id: str, emotion: str) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
        (obs_id, "backfill test", "unknown", "conversation", emotion, AGENT_SELF_ID),
    )


def _pad(cur, obs_id: str) -> tuple[float, float, float, float]:
    cur.execute(
        "SELECT emotion_p, emotion_pn, emotion_a, emotion_dom FROM observations WHERE id = %s",
        (obs_id,),
    )
    r = cur.fetchone()
    return (r["emotion_p"], r["emotion_pn"], r["emotion_a"], r["emotion_dom"])


# ── 1. 写像が既存行へ適用される ─────────────────────────────────────────────
def test_backfill_applies_mapping() -> None:
    ids = {label: str(uuid.uuid4()) for label in ("happy", "sad", "moved")}
    conn = _pg_conn()
    _run_columns_migration(conn)
    with conn.cursor() as cur:
        for label, oid in ids.items():
            _insert(cur, oid, label)
    conn.commit()
    _run_backfill_migration(conn)
    with conn.cursor() as cur:
        assert _pad(cur, ids["happy"]) == (0.80, 0.15, 0.55, 0.60)
        assert _pad(cur, ids["sad"]) == (0.20, 0.75, 0.25, 0.30)
        assert _pad(cur, ids["moved"]) == (0.75, 0.50, 0.60, 0.45)
    conn.close()


# ── 2. 表に無いラベルは既定0.5のまま（既知ラベルだけ触る） ───────────────────
def test_backfill_leaves_unknown_label_at_default() -> None:
    oid = str(uuid.uuid4())
    conn = _pg_conn()
    _run_columns_migration(conn)
    with conn.cursor() as cur:
        _insert(cur, oid, "bogus")
    conn.commit()
    _run_backfill_migration(conn)
    with conn.cursor() as cur:
        assert _pad(cur, oid) == (0.5, 0.5, 0.5, 0.5)
    conn.close()


# ── 3. neutral も明示的に更新対象（(b)・既定と同値） ────────────────────────
def test_backfill_updates_neutral_explicitly() -> None:
    oid = str(uuid.uuid4())
    conn = _pg_conn()
    _run_columns_migration(conn)
    with conn.cursor() as cur:
        _insert(cur, oid, "neutral")
    conn.commit()
    mod = _run_backfill_migration(conn)
    with conn.cursor() as cur:
        assert _pad(cur, oid) == (0.5, 0.5, 0.5, 0.5)
    conn.close()
    assert "neutral" in mod._LABEL_PAD


# ── 4. 写像表の網羅（valid からのドリフト防止） ─────────────────────────────
def test_mapping_table_covers_expected_labels() -> None:
    conn = _pg_conn()
    _run_columns_migration(conn)
    mod = _run_backfill_migration(conn)
    conn.close()
    table = mod._LABEL_PAD
    assert set(table.keys()) == _EXPECTED_LABELS
    for label, pad in table.items():
        assert len(pad) == 4, label
        assert all(0.0 <= x <= 1.0 for x in pad), label
