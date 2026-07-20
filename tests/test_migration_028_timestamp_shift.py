"""Tests for migration 028（既存観測の9時間ずれの補正）.

既存行は JST の壁掛け時計が UTC として保存されており、実時刻より9時間先にある。
書き込み側を tz-aware に直したので、既存行も同じ時計へ寄せる。二度当てると余分に
ずれるため、冪等であることも見る。
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"
_MIGRATION = "2026-07-20-028_fix_observation_timestamp_tz.py"


def _pg():
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


def _insert(conn, content: str, ts_sql: str) -> str:
    obs_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, person_id) "
            f"VALUES (%s, %s, {ts_sql}, %s, %s, %s)",
            (obs_id, content, "会話", "conversation", DEFAULT_PERSON_ID),
        )
    return obs_id


def _drift_hours(conn, obs_id: str) -> float:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXTRACT(EPOCH FROM (timestamp - now())) / 3600 AS h "
            "FROM observations WHERE id = %s",
            (obs_id,),
        )
        return float(cur.fetchone()["h"])


def test_shifts_future_rows_back_by_nine_hours() -> None:
    """9時間先の行が現在時刻へ寄る。"""
    conn = _pg()
    obs_id = _insert(conn, f"tz shift {uuid.uuid4()}", "now() + interval '9 hours'")
    assert _drift_hours(conn, obs_id) > 8.5, "前提：ずれた行を置けている"

    _run_migration(conn)

    assert abs(_drift_hours(conn, obs_id)) < 0.1, "補正後も現在時刻からずれている"
    conn.close()


def test_migration_is_idempotent() -> None:
    """二度当てても余分にずれない（未来の行が無ければ何もしない）。"""
    conn = _pg()
    obs_id = _insert(conn, f"tz idem {uuid.uuid4()}", "now() + interval '9 hours'")
    _run_migration(conn)
    after_first = _drift_hours(conn, obs_id)

    _run_migration(conn)
    after_second = _drift_hours(conn, obs_id)

    assert abs(after_second - after_first) < 0.1, "二度目の適用で余分にずれた"
    conn.close()


def test_last_recalled_at_is_left_alone() -> None:
    """last_recalled_at は SQL の now() で書かれており正しいので触らない。"""
    conn = _pg()
    obs_id = _insert(conn, f"tz keep {uuid.uuid4()}", "now() + interval '9 hours'")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE observations SET last_recalled_at = now() WHERE id = %s", (obs_id,)
        )
        cur.execute(
            "SELECT last_recalled_at FROM observations WHERE id = %s", (obs_id,)
        )
        before = cur.fetchone()["last_recalled_at"]

    _run_migration(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_recalled_at FROM observations WHERE id = %s", (obs_id,)
        )
        after = cur.fetchone()["last_recalled_at"]
    assert after == before, "last_recalled_at が動いている"
    conn.close()
