"""063（旧 15 欲求の状態 `agent_state.desires` を消す・環-d・2026-09-15）。

書き手 `DesireSystem`（`desires.py`）はこの日に撤去した。欲求は 5 軸（`drive_register`・
`core/drive_dynamics`・`agent_state.drive5`）が担う。
"""

from __future__ import annotations

import os
from importlib import util

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = "migration/2026-09-15-063_drop_desires_state.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _module():
    spec = util.spec_from_file_location("m063", _PATH)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _has_row() -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM agent_state WHERE state_key = 'desires'")
        return cur.fetchone() is not None


def test_upgrade_removes_the_row_and_is_idempotent():
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES ('desires', '{}', now()) "
            "ON CONFLICT (state_key) DO NOTHING"
        )
    assert _has_row()
    _module().upgrade(conn)
    assert not _has_row()
    _module().upgrade(conn)
    assert not _has_row()
    conn.close()
