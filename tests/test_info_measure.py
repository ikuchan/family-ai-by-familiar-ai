"""使われる情報量 $I=\\sum b_i u_i$ と減り $\\Delta$（`出来事を畳む` §1・§4・記-a-ろ-ろ）。純関数。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from familiar_agent.core import info_measure as im

NOW = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)


def test_bits_are_estimated_from_the_character_count() -> None:
    assert im.bits("") == 0.0
    assert im.bits("あいうえお") == 5 * im.BITS_PER_CHAR
    assert im.bits_of_chars(100) == 100 * im.BITS_PER_CHAR


def test_usability_is_the_two_axis_merit_normalised_to_one() -> None:
    # 今日書かれ（t=1）根づきが最大（g=2）なら 1。
    assert im.usability(age_days=0.0, g0=1.0, n=10**6, hl=10.0, w_t=1.0, w_g=1.5) == 1.0
    # 半減期ぶん経てば t は 0.5。n=0・g0=1.0 なら g=1.0（範囲 0〜2 の中央）。
    u = im.usability(age_days=10.0, g0=1.0, n=0, hl=10.0, w_t=1.0, w_g=1.5)
    m = (1.0 * 0.5 + 1.5 * 1.0) / 2.5
    m_max = (1.0 * 1.0 + 1.5 * 2.0) / 2.5
    assert abs(u - m / m_max) < 1e-9
    # 古くなるほど・根づきが低いほど小さい（単調）。
    assert im.usability(age_days=30.0, g0=0.5, n=0, hl=10.0, w_t=1.0, w_g=1.5) < im.usability(
        age_days=1.0, g0=0.5, n=0, hl=10.0, w_t=1.0, w_g=1.5
    )
    assert im.usability(age_days=1.0, g0=0.5, n=0, hl=10.0, w_t=1.0, w_g=1.5) < im.usability(
        age_days=1.0, g0=0.5, n=3, hl=10.0, w_t=1.0, w_g=1.5
    )


def test_total_sums_bits_times_usability_over_rows() -> None:
    rows = [
        {"chars": 100, "timestamp": NOW, "groundedness_g0": 1.0, "groundedness_n": 10**6},
        {
            "chars": 50,
            "timestamp": NOW - timedelta(days=10),
            "groundedness_g0": 1.0,
            "groundedness_n": 0,
        },
    ]
    t = im.total(rows, now=NOW, hl=10.0, w_t=1.0, w_g=1.5)
    u2 = im.usability(age_days=10.0, g0=1.0, n=0, hl=10.0, w_t=1.0, w_g=1.5)
    assert abs(t - (100 * im.BITS_PER_CHAR * 1.0 + 50 * im.BITS_PER_CHAR * u2)) < 1e-6
    assert im.total([], now=NOW, hl=10.0, w_t=1.0, w_g=1.5) == 0.0


def test_delta_grows_with_the_log_of_the_excess_and_is_zero_below_target() -> None:
    target = 2.0**17
    assert im.delta_for(0.0, target) == 0
    assert im.delta_for(target * 0.5, target) == 0
    assert im.delta_for(target, target) == 0
    assert im.delta_for(target * 1.5, target) == 1
    assert im.delta_for(target * 2.0, target) == 1
    assert im.delta_for(target * 2.1, target) == 2
    assert im.delta_for(target * 5.0, target) == 3
    assert im.delta_for(target, 0.0) == 0  # 目標が壊れていても落ちない
    assert math.isfinite(im.delta_for(1e30, target))
