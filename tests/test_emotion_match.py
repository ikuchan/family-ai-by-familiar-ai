"""感情一致 e（e 軸）の純関数テスト（Phase 2 P-3・スライス1・未接続）。

課題5 v0.23 で確定したガウシアン e=exp(-D²/(2σ²)) を、ロジット空間の
軸重み付き PAD 距離 D で計算する `_emotion_match` の単体テスト。DB 不要。
"""

from __future__ import annotations

import math

import pytest

from familiar_agent.tools.memory import _emotion_match


_NEUTRAL = (0.5, 0.5, 0.5, 0.5)


# ── 1. 完全一致 → e == 1.0 ──────────────────────────────────────────────────
def test_identical_pad_returns_one() -> None:
    for pad in [_NEUTRAL, (0.7, 0.3, 0.6, 0.4), (0.1, 0.9, 0.5, 0.2)]:
        assert _emotion_match(pad, pad) == pytest.approx(1.0)


# ── 2. 単調性：中立 mood から遠い観測ほど e が小さい ────────────────────────
def test_monotonic_decreasing_with_distance() -> None:
    near = _emotion_match((0.6, 0.5, 0.5, 0.5), _NEUTRAL)
    far = _emotion_match((0.8, 0.5, 0.5, 0.5), _NEUTRAL)
    farther = _emotion_match((0.95, 0.5, 0.5, 0.5), _NEUTRAL)
    assert near > far > farther


# ── 3. 範囲：0 < e <= 1 ──────────────────────────────────────────────────────
def test_range_is_open_zero_to_one() -> None:
    for obs in [_NEUTRAL, (0.9, 0.1, 0.8, 0.2), (0.0, 1.0, 0.0, 1.0)]:
        e = _emotion_match(obs, _NEUTRAL)
        assert 0.0 < e <= 1.0


# ── 4. 対称性 ────────────────────────────────────────────────────────────────
def test_symmetric() -> None:
    a = (0.7, 0.3, 0.6, 0.4)
    b = (0.4, 0.8, 0.2, 0.5)
    assert _emotion_match(a, b) == pytest.approx(_emotion_match(b, a))


# ── 5. 端クランプ：0.0 / 1.0 でも inf/nan にならず有限で (0,1] ────────────────
def test_edge_values_are_finite() -> None:
    e = _emotion_match((0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 0.0))
    assert math.isfinite(e)
    assert 0.0 < e <= 1.0


# ── 6. 軸重み：差のある軸の λ を上げると e が下がる ─────────────────────────
def test_axis_weight_lowers_match_for_differing_axis() -> None:
    obs = (0.9, 0.5, 0.5, 0.5)  # P 軸だけ中立から離れている
    base = _emotion_match(obs, _NEUTRAL, lambdas=(1.0, 1.0, 1.0, 1.0))
    heavier = _emotion_match(obs, _NEUTRAL, lambdas=(3.0, 1.0, 1.0, 1.0))
    assert heavier < base


# ── 7. σ：上げると寛容（e が 1 に近づく） ───────────────────────────────────
def test_larger_sigma_is_more_tolerant() -> None:
    obs = (0.85, 0.5, 0.5, 0.5)
    tight = _emotion_match(obs, _NEUTRAL, sigma=0.5)
    loose = _emotion_match(obs, _NEUTRAL, sigma=2.0)
    assert loose > tight
