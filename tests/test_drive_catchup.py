"""案B：起動時キャッチアップ。停止中の経過（now − drive5.updated_at）を初回 tick に積む。

cap は設けず accumulate の [0,1] クリップ任せ。mood は起動時 snapshot 近似。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest

from familiar_agent.drive_register import (
    AiDrivers,
    catchup_dt,
    load_drives_with_updated_at,
    save_drives,
)


# ── catchup_dt（純関数・停止秒数） ───────────────────────────────────────────

def test_catchup_dt_none_is_zero():
    assert catchup_dt(None, 100.0) == 0.0


def test_catchup_dt_elapsed():
    updated = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    now = datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    assert catchup_dt(updated, now.timestamp()) == pytest.approx(10.0, abs=1e-6)


def test_catchup_dt_future_clamped_to_zero():
    updated = datetime(2020, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    now = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert catchup_dt(updated, now.timestamp()) == 0.0


# ── load_drives_with_updated_at（実 DB・更新時刻を返す） ──────────────────────

def test_load_drives_with_updated_at_roundtrip():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    save_drives(conn, AiDrivers(seeking=0.3))
    drives, updated = load_drives_with_updated_at(conn)
    conn.close()
    assert drives.seeking == pytest.approx(0.3)
    assert updated is not None  # 更新時刻が取れる


def test_offtime_catchup_accumulates():
    """updated_at を過去へ倒すと、その経過分が初回 tick で積み上がる。"""
    from familiar_agent.core.drive_dynamics import tick
    from familiar_agent.drive_register import DRIVE_STATE_KEY
    from familiar_agent.mood_register import MoodPAD

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    save_drives(conn, AiDrivers())  # 全0
    # updated_at を10分前へ倒す
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE agent_state SET updated_at = %s WHERE state_key = %s",
            (past, DRIVE_STATE_KEY),
        )
    drives, updated = load_drives_with_updated_at(conn)
    conn.close()

    dt = catchup_dt(updated, datetime.now(timezone.utc).timestamp())
    assert dt == pytest.approx(600.0, abs=5.0)  # ≈10分
    d1, firing = tick(drives, MoodPAD(), dt=dt)  # 中立 mood で経過分を積む
    # 10分・中立 SEEKING（b=0.20）は閾値到達→発火（accumulate が 1.0 にクリップ）
    assert firing.seeking is True
