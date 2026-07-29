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
    """1 tick の増分が、深夜は昼間の `mult_quiet` 倍になる（REST 以外）。

    REST だけは別の倍率で募る（`test_rest_is_not_slowed_at_night_but_hastened`）。
    """
    cfg = DriveConfig()
    day = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DAYTIME))
    night = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DEEP_NIGHT))
    for axis in ("seeking", "bond", "safety", "esteem"):
        assert getattr(night, axis) == pytest.approx(
            getattr(day, axis) * cfg.mult_quiet, rel=1e-9), axis


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


# ── 軸別の深夜倍率（REST は夜に募る）─────────────────────────────────────

def test_rest_is_not_slowed_at_night_but_hastened():
    """深夜の倍率は軸ごとに違う。REST だけは抑えず、逆に募らせる。

    設計（`設計詳細：発火・mood 機構` §82）は「REST の募りは別途バイアス＋時間帯倍率
    （夜高い）」と定める。全軸へ一律に 0.083 を掛けると、これと正反対になる。
    """
    cfg = effective_drive_cfg(DriveConfig(), now=_DEEP_NIGHT)
    assert cfg.mult_for("rest") > 1.0            # 夜は募る
    assert cfg.mult_for("seeking") == pytest.approx(DriveConfig().mult_quiet)


def test_daytime_multipliers_are_one_for_every_axis():
    cfg = effective_drive_cfg(DriveConfig(), now=_DAYTIME)
    for axis in ("seeking", "rest", "bond", "safety", "esteem"):
        assert cfg.mult_for(axis) == pytest.approx(1.0), axis


def test_rest_fires_exactly_once_during_the_quiet_window():
    """中立 mood の深夜（8時間）に、REST がちょうど1回発火する量まで溜まる。

    2回は起きない（起きるには 2.0 が要る）。深夜の開始時に drive が 0 でも届くように、
    閾値に対して余裕を持たせてある。
    """
    cfg = effective_drive_cfg(DriveConfig(), now=_DEEP_NIGHT)
    night_sec = 8 * 3600
    accumulated = cfg.rate * cfg.mult_for("rest") * cfg.learn * cfg.bias_rest * night_sec
    assert accumulated >= cfg.theta_fire          # 必ず1回は起きる
    assert accumulated < 2 * cfg.theta_fire       # 2回は起きない


def test_accumulation_uses_the_per_axis_multiplier():
    """蓄積が軸ごとの倍率を使う（REST だけ速く、他は遅く）。"""
    cfg = DriveConfig()
    day = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DAYTIME))
    night = accumulate(AiDrivers(), MoodPAD(), cfg=effective_drive_cfg(cfg, now=_DEEP_NIGHT))
    assert night.rest > day.rest                  # REST は夜のほうが速い
    assert night.seeking < day.seeking            # 探索は夜のほうが遅い
