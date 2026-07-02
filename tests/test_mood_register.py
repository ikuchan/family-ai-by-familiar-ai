"""Tests for the mood PAD register (Phase 1 B-1, thin vertical slice).

mood_register is a new, unconnected module: a PAD-axis vessel that decays
toward the midpoint (0.5) with a 600s half-life and persists via agent_state.
Not wired to appraisal/recall/_mood/_decayed_mood/AffectiveState in this step.
"""

from __future__ import annotations

import psycopg2

from familiar_agent.mood_register import MoodPAD, decay_to_rest, load_mood, save_mood


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _db_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    return conn


# ── decay direction: from above (0.5 < x) settles downward ──────────────────

def test_decay_neutral_stays_neutral() -> None:
    m = MoodPAD()  # all axes 0.5
    out = decay_to_rest(m, 600.0)
    assert (out.p, out.pn, out.a, out.dom) == (0.5, 0.5, 0.5, 0.5)


def test_decay_converges_from_above() -> None:
    out = decay_to_rest(MoodPAD(p=0.9), 600.0)
    assert abs(out.p - 0.7) < 1e-9  # 0.5 + (0.9-0.5)*0.5 after one half-life


# ── decay direction: from below (x < 0.5) settles upward ────────────────────
# Falsification: a floor(0)-decay implementation would push this toward 0.05,
# not 0.3 — this test only passes for a midpoint-rest implementation.

def test_decay_converges_from_below() -> None:
    out = decay_to_rest(MoodPAD(pn=0.1), 600.0)
    assert abs(out.pn - 0.3) < 1e-9  # 0.5 + (0.1-0.5)*0.5 after one half-life


def test_decay_axes_independent_and_asymptote() -> None:
    out = decay_to_rest(MoodPAD(p=0.9, pn=0.1, a=0.8, dom=0.2), 6000.0)
    for v in (out.p, out.pn, out.a, out.dom):
        assert abs(v - 0.5) < 1e-3  # after many half-lives, all axes settle at rest


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

    assert (got.p, got.pn, got.a, got.dom) == (0.5, 0.5, 0.5, 0.5)
