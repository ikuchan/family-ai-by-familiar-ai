"""Tests for migration 028（既存観測の9時間ずれの補正）.

既存行は JST の壁掛け時計が UTC として保存されており、実時刻より9時間先にある。
書き込み側を tz-aware に直したので、既存行も同じ時計へ寄せる。

二重適用は二段で防ぐ。主たる保証はランナー（`schema_migrations` で適用済みを記録）、
保険が移行本体の前提条件（壊れている間だけ成り立つ `max(timestamp) > now()`）。
ここでは両方を見る。前提条件には限界があり（相対的なずらしは補正後と元から正しい
データを見分けられない）、壊れた行と正しい行を混在させた状態は本番では起きないので、
テストでもその状態を作らない。
"""

from __future__ import annotations

import os

import importlib.util
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID


_DB_URL = os.environ["DATABASE_URL"]
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


def test_second_direct_run_does_nothing() -> None:
    """本体を直接二度呼んでも2回目は何もしない（前提条件の保険）。"""
    conn = _pg()
    obs_id = _insert(conn, f"tz idem {uuid.uuid4()}", "now() + interval '9 hours'")
    _run_migration(conn)
    after_first = _drift_hours(conn, obs_id)

    _run_migration(conn)
    after_second = _drift_hours(conn, obs_id)

    assert abs(after_second - after_first) < 0.1, "二度目の適用で余分にずれた"
    conn.close()


def test_runner_applies_it_only_once() -> None:
    """ランナー経由でも二度当たらない（主たる保証は schema_migrations）。"""
    from familiar_agent.db_migrations import apply_migrations, default_migration_dir

    conn = _pg()
    obs_id = _insert(conn, f"tz runner {uuid.uuid4()}", "now() + interval '9 hours'")

    # ランナーは row[0] で読むので、本番と同じ素の接続を渡す（db.py の conn()）。
    runner_conn = psycopg2.connect(_DB_URL)
    runner_conn.autocommit = True
    mig_dir = default_migration_dir()
    try:
        apply_migrations(runner_conn, migration_dir=mig_dir)
        after_first = _drift_hours(conn, obs_id)
        apply_migrations(runner_conn, migration_dir=mig_dir)
        after_second = _drift_hours(conn, obs_id)
    finally:
        runner_conn.close()

    assert abs(after_second - after_first) < 0.1, "ランナー経由で二度当たった"
    conn.close()


