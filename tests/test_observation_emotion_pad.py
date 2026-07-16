"""Tests for observations の感情 PAD 列（案B・書き込み PAD 化 W1a・schema器）。

課題5 v0.23：観測 emotion を4軸 PAD（P/Pn/A/Dom・各 [0,1]・中立0.5）で持つ。
この段（W1a）は列を追加するだけで、既存行・新規行とも既定0.5。評価器・スコア・
recall は無変更で列は誰も読まない（外部挙動不変）。ラベル→PAD 写像（W1b）と
評価器の PAD 出力（W2）は後続。
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.errors
import psycopg2.extras
import pytest

from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"

_PAD_COLUMNS = {"emotion_p", "emotion_pn", "emotion_a", "emotion_dom"}


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _run_migration(conn) -> None:
    migration_path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-07-16-024_observation_emotion_pad.py"
    )
    spec = importlib.util.spec_from_file_location("observation_emotion_pad_migration", migration_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def _insert_observation(cur, obs_id: str) -> None:
    """PAD 列を指定しない既存経路と同型の INSERT。"""
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, NOW(), %s, %s, %s, %s)",
        (obs_id, "emotion pad test", "unknown", "conversation", "neutral", AGENT_SELF_ID),
    )


# ── 1. 4列が揃う ────────────────────────────────────────────────────────────
def test_migration_adds_pad_columns() -> None:
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'observations'
              AND column_name IN ('emotion_p', 'emotion_pn', 'emotion_a', 'emotion_dom')
        """)
        cols = {r["column_name"] for r in cur.fetchall()}
    conn.close()
    assert cols == _PAD_COLUMNS


# ── 2. 既定0.5：PAD 列を指定しない INSERT で4軸とも 0.5 ────────────────────────
def test_pad_defaults_to_neutral_half() -> None:
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        _insert_observation(cur, obs_id)
        cur.execute(
            "SELECT emotion_p, emotion_pn, emotion_a, emotion_dom "
            "FROM observations WHERE id = %s",
            (obs_id,),
        )
        row = cur.fetchone()
    conn.close()
    assert (row["emotion_p"], row["emotion_pn"], row["emotion_a"], row["emotion_dom"]) == (
        0.5, 0.5, 0.5, 0.5,
    )


# ── 3. CHECK 下限：範囲外の UPDATE が弾かれる ───────────────────────────────
def test_check_rejects_below_zero() -> None:
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        _insert_observation(cur, obs_id)
    conn.commit()
    with pytest.raises(psycopg2.errors.CheckViolation):
        with conn.cursor() as cur:
            cur.execute("UPDATE observations SET emotion_p = -0.1 WHERE id = %s", (obs_id,))
    conn.rollback()
    conn.close()


# ── 4. CHECK 上限：範囲外の UPDATE が弾かれる ───────────────────────────────
def test_check_rejects_above_one() -> None:
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    _run_migration(conn)
    with conn.cursor() as cur:
        _insert_observation(cur, obs_id)
    conn.commit()
    with pytest.raises(psycopg2.errors.CheckViolation):
        with conn.cursor() as cur:
            cur.execute("UPDATE observations SET emotion_a = 1.5 WHERE id = %s", (obs_id,))
    conn.rollback()
    conn.close()
