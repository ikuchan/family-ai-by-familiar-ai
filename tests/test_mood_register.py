"""Tests for the mood PAD register (Phase 1 B-1, thin vertical slice).

mood_register is a new, unconnected module: a PAD-axis vessel that decays
toward the midpoint (0.5) with a 600s half-life and persists via agent_state.
Not wired to appraisal/recall/_mood/_decayed_mood/AffectiveState in this step.
"""

from __future__ import annotations

import os

import psycopg2

from familiar_agent.mood_register import MoodPAD, decay_to_rest, load_mood, save_mood


_DB_URL = os.environ["DATABASE_URL"]


def _db_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    return conn


# ── 減衰の向き：戻り先より上からは下がる ────────────────────────────────────
# 戻り先は軸ごとに違う（案A）。P と Pn は 0.10、A と Dom は 0.50 である。

def test_decay_at_rest_stays_at_rest() -> None:
    out = decay_to_rest(MoodPAD(), 600.0)
    assert (out.p, out.pn, out.a, out.dom) == (0.10, 0.10, 0.50, 0.50)


def test_decay_converges_from_above() -> None:
    out = decay_to_rest(MoodPAD(p=0.9), 600.0)
    assert abs(out.p - 0.5) < 1e-9  # 0.10 + (0.9-0.10)*0.5 ＝ 半減期1つぶん


# ── 減衰の向き：戻り先より下からは上がる ────────────────────────────────────
# 反証：0 へ落とす実装なら Dom は 0.10 へ向かう。中点へ戻る実装でしか 0.35 にならない。

def test_decay_converges_from_below() -> None:
    out = decay_to_rest(MoodPAD(dom=0.2), 600.0)
    assert abs(out.dom - 0.35) < 1e-9  # 0.50 + (0.2-0.50)*0.5


def test_decay_axes_settle_at_their_own_rest() -> None:
    out = decay_to_rest(MoodPAD(p=0.9, pn=0.1, a=0.8, dom=0.2), 6000.0)
    assert abs(out.p - 0.10) < 1e-3
    assert abs(out.pn - 0.10) < 1e-3
    assert abs(out.a - 0.50) < 1e-3
    assert abs(out.dom - 0.50) < 1e-3


def test_clip_range() -> None:
    m = MoodPAD(p=1.5, pn=-0.3).clipped()
    assert m.p == 1.0 and m.pn == 0.0


# ── persistence via agent_state ──────────────────────────────────────────────

def test_save_load_round_trip() -> None:
    conn = _db_conn()
    save_mood(conn, MoodPAD(p=0.7, pn=0.2, a=0.6, dom=0.4))
    got = load_mood(conn)
    conn.close()

    assert (got.p, got.pn, got.a, got.dom) == (0.7, 0.2, 0.6, 0.4)


def test_load_default_when_absent() -> None:
    conn = _db_conn()
    got = load_mood(conn)  # mood_pad key not present
    conn.close()

    assert (got.p, got.pn, got.a, got.dom) == (0.10, 0.10, 0.50, 0.50)
