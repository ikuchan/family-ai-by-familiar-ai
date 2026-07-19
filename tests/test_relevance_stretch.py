"""Tests for the relevance stretch `_stretch_relevance` (Phase 2 スライス3).

r 軸は生コサインをそのまま使わず、固定係数の min-max 伸長
r = clip((cos - c_lo) / (c_hi - c_lo), 0, 1) を通す（課題5 v0.24 D 節）。
現行の確定値 c_lo=0.0 / c_hi=1.0 では恒等になるが、両者は Config で可変なので
r の経路を一本に保つために式を通す。DB 非依存の単体。
"""

from __future__ import annotations

from familiar_agent.tools.memory import _stretch_relevance


def test_identity_with_confirmed_coefficients() -> None:
    """c_lo=0.0・c_hi=1.0（課題5 v0.24 の確定値）では恒等。"""
    for cos in (0.0, 0.13, 0.5, 0.87, 1.0):
        assert abs(_stretch_relevance(cos, c_lo=0.0, c_hi=1.0) - cos) < 1e-12


def test_clips_below_c_lo_to_zero() -> None:
    """c_lo 以下は 0（中心化後は負のコサインが出る）。"""
    assert _stretch_relevance(-0.3, c_lo=0.0, c_hi=1.0) == 0.0
    assert _stretch_relevance(0.2, c_lo=0.25, c_hi=0.5) == 0.0


def test_clips_above_c_hi_to_one() -> None:
    """c_hi 以上は 1。"""
    assert _stretch_relevance(0.6, c_lo=0.25, c_hi=0.5) == 1.0


def test_stretches_between_the_coefficients() -> None:
    """区間内は線形に [0,1] へ伸長される。"""
    # 0.25〜0.75 の中点 0.5 は 0.5 へ
    assert abs(_stretch_relevance(0.5, c_lo=0.25, c_hi=0.75) - 0.5) < 1e-12
    # 区間が狭いほど同じコサイン差が大きく開く
    assert _stretch_relevance(0.4, c_lo=0.25, c_hi=0.5) > _stretch_relevance(
        0.4, c_lo=0.25, c_hi=0.75
    )


def test_degenerate_range_becomes_step_without_zero_division() -> None:
    """c_hi <= c_lo では段階を作れないのでステップ関数へ退化する（0除算しない）。"""
    assert _stretch_relevance(0.6, c_lo=0.5, c_hi=0.5) == 1.0
    assert _stretch_relevance(0.4, c_lo=0.5, c_hi=0.5) == 0.0
    assert _stretch_relevance(0.9, c_lo=0.8, c_hi=0.3) == 1.0
