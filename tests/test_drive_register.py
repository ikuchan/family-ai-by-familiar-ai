"""Tests for the new-5 drive register (Phase 1 B-2, vessel only).

drive_register is a new, unconnected module: a SEEKING/REST/BOND/SAFETY/ESTEEM
vessel persisted via agent_state. Not wired to DesireSystem/as_coalition/agent.py
in this step; accumulation, discharge, and mood modulation (dynamics) are later work.
"""

from __future__ import annotations

import psycopg2

from familiar_agent.drive_register import AiDrivers, load_drives, save_drives


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"


def _db_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    return conn


# ── still (default) state ────────────────────────────────────────────────────

def test_default_is_still() -> None:
    d = AiDrivers()
    assert (d.seeking, d.rest, d.bond, d.safety, d.esteem) == (0.0, 0.0, 0.0, 0.0, 0.0)


# ── clip range ────────────────────────────────────────────────────────────────

def test_clip_range() -> None:
    d = AiDrivers(seeking=1.5, rest=-0.2, bond=0.5).clipped()
    assert d.seeking == 1.0 and d.rest == 0.0 and d.bond == 0.5


# ── JSON round trip ───────────────────────────────────────────────────────────

def test_json_round_trip() -> None:
    d = AiDrivers(seeking=0.3, rest=0.1, bond=0.7, safety=0.2, esteem=0.4)
    assert AiDrivers.from_json_dict(d.to_json_dict()) == d


# ── agent_state persistence round trip ───────────────────────────────────────

def test_save_load_round_trip() -> None:
    conn = _db_conn()
    save_drives(conn, AiDrivers(seeking=0.3, rest=0.1, bond=0.7, safety=0.2, esteem=0.4))
    got = load_drives(conn)
    conn.close()

    assert (got.seeking, got.rest, got.bond, got.safety, got.esteem) == (0.3, 0.1, 0.7, 0.2, 0.4)


# ── default when the drive5 key is absent ────────────────────────────────────

def test_load_default_when_absent() -> None:
    conn = _db_conn()
    got = load_drives(conn)  # drive5 key not present
    conn.close()

    assert (got.seeking, got.rest, got.bond, got.safety, got.esteem) == (0.0, 0.0, 0.0, 0.0, 0.0)
