"""Tests for the (a0, n) activation representation (Phase 1 A-3-1)."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.tools.memory import _derive_activation
from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _run_migration(conn) -> None:
    migration_path = (
        Path(__file__).parent.parent
        / "migration"
        / "2026-07-02-021_activation_a0_n.py"
    )
    spec = importlib.util.spec_from_file_location("activation_a0_n_migration", migration_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


# ── 1. n=0 returns a0 (logit/logistic are inverses) ─────────────────────────

def test_derive_activation_n_zero_returns_a0() -> None:
    for a0 in (0.5, 0.75, 1.5):
        assert _derive_activation(a0, 0) == pytest.approx(a0)


# ── 2. monotonic in n ────────────────────────────────────────────────────────

def test_derive_activation_monotonic_in_n() -> None:
    a0 = 1.0
    a_minus = _derive_activation(a0, -2)
    a_zero = _derive_activation(a0, 0)
    a_plus = _derive_activation(a0, 2)

    assert a_minus < a_zero < a_plus


# ── 3. asymptotes toward floor/C at extremes ────────────────────────────────

def test_derive_activation_approaches_bounds() -> None:
    a0 = 1.0
    high = _derive_activation(a0, 50)
    low = _derive_activation(a0, -50)

    assert high < 2.0
    assert high > 1.99
    assert low > 0.0
    assert low < 0.01


# ── 4. +1 then -1 returns to the original value (round-trip) ────────────────

def test_derive_activation_plus_minus_one_round_trip() -> None:
    a0 = 0.6
    up = _derive_activation(a0, 1)
    # deriving from a0 directly with n=+1 then n=-1 from the same a0 baseline
    # (symmetry check around n=0, not a chained re-derivation)
    down = _derive_activation(a0, -1)
    mid = _derive_activation(a0, 0)

    assert (up - mid) > 0
    assert (mid - down) > 0
    assert abs((up - mid) - (mid - down)) < 0.2  # roughly symmetric near center


# ── 5. a0=0.75, step=0.33 → 評価5回で実用上限1.5に到達 ────────────────────────

def test_derive_activation_reaches_practical_limit_at_five() -> None:
    a4 = _derive_activation(0.75, 4)
    a5 = _derive_activation(0.75, 5)
    assert a4 < 1.5 <= a5   # 4回では1.5未満、5回で1.5到達
    assert a5 < 1.6         # ハード上限C=2にはまだ遠い（緩んで育つ）


# ── 6. migration adds the two columns ────────────────────────────────────────

def test_migration_adds_activation_columns() -> None:
    conn = _pg_conn()
    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'observations'
              AND column_name IN ('activation_a0', 'activation_n')
        """)
        cols = {r["column_name"] for r in cur.fetchall()}
    conn.close()

    assert cols == {"activation_a0", "activation_n"}


# ── 7. migration copies importance into activation_a0, n defaults to 0 ──────

def test_migration_migrates_importance_into_activation_a0() -> None:
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
            "person_id, importance) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)",
            (obs_id, "activation migration test", "unknown", "curiosity", "neutral",
             AGENT_SELF_ID, 0.42),
        )
    conn.commit()

    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT importance, activation_a0, activation_n FROM observations WHERE id = %s",
            (obs_id,),
        )
        row = cur.fetchone()
    conn.close()

    assert row["activation_a0"] == row["importance"]
    assert row["activation_n"] == 0
