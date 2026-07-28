"""深夜の時間帯倍率（#13）：静穏時間は Drive の蓄積を遅くする。

設計式 `drive_i += rate·mult(t)·learn·g_D·dt` の `mult(t)` を、静穏時間だけ下げる。
時計を見るのは T の役なので、判定は `loop.tonic` に置く（`core.drive_dynamics` は
時計を持たない純関数）。倍率は全欲求へ一律にかかる（設計式に軸の添字が無い）。
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from datetime import datetime

import psycopg2
import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core.drive_dynamics import accumulate
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.tonic import effective_drive_cfg
from familiar_agent.mood_register import MoodPAD

# 静穏時間の既定は 23〜7（`QUIET_HOURS_START`／`END`）。その内と外を1点ずつ取る。
_DEEP_NIGHT = datetime(2026, 7, 29, 2, 0, 0)
_DAYTIME = datetime(2026, 7, 29, 12, 0, 0)


def test_quiet_hours_use_the_reduced_multiplier():
    cfg = DriveConfig()
    assert effective_drive_cfg(cfg, now=_DEEP_NIGHT).mult == pytest.approx(cfg.mult_quiet)


def test_daytime_keeps_the_normal_multiplier():
    cfg = DriveConfig()
    assert effective_drive_cfg(cfg, now=_DAYTIME).mult == pytest.approx(cfg.mult)


def test_accumulation_slows_by_the_quiet_multiplier():
    """1 tick の増分が、深夜は昼間の `mult_quiet` 倍になる（全欲求一律）。"""
    cfg = DriveConfig()
    day = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DAYTIME))
    night = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DEEP_NIGHT))
    assert night.seeking == pytest.approx(day.seeking * cfg.mult_quiet, rel=1e-9)
    assert night.rest == pytest.approx(day.rest * cfg.mult_quiet, rel=1e-9)


def test_seeking_fires_about_once_an_hour_at_night():
    """中立 mood の深夜、探索が 0 から Θ_fire へ至るまでが 3600 秒程度になる。

    これが #13 の要求そのものである（探索の発火を1時間に1回程度に抑える）。放電は
    全放電（q = 1−ε）なので、0 から閾値までの時間がそのまま発火の周期になる。
    """
    cfg = effective_drive_cfg(DriveConfig(), now=_DEEP_NIGHT)
    # 中立 mood では g_seeking = bias_seeking。増分は毎秒一定なので割り算で出る。
    per_second = cfg.rate * cfg.mult * cfg.learn * cfg.bias_seeking
    seconds_to_fire = cfg.theta_fire / per_second
    assert seconds_to_fire == pytest.approx(3600.0, rel=0.05)


def test_step_drives_applies_the_time_of_day_multiplier():
    """T の tick が、時間帯倍率を通した設定で蓄積する（倍率が実際に効く配線）。

    倍率を 0 に差し替えれば、1分回しても drive は増えない。時刻ではなく倍率そのもので
    判定するので、テストを走らせる時刻に左右されない。
    """
    from familiar_agent.drive_register import save_drives
    from familiar_agent.loop import tonic as tonic_module

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    save_drives(conn, AiDrivers())          # 全軸 0 から始める
    conn.close()

    original = tonic_module.effective_drive_cfg
    tonic_module.effective_drive_cfg = lambda cfg, now=None: replace(cfg, mult=0.0)
    try:
        _, accumulated = asyncio.run(tonic_module.step_drives(60.0))
    finally:
        tonic_module.effective_drive_cfg = original
    assert accumulated.seeking == pytest.approx(0.0, abs=1e-12)
