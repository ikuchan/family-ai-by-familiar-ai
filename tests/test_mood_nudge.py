"""Tests for mood の nudge と N_PAD 計算（mood-a・未接続）。

課題5：N_PAD＝W 全 MI の PAD の 根づき加重平均＋自己認識 MI のフラット項
(0.5,0.5,0.5,0.5)・重み C=2.0。nudge は A_M←max(A_M,A_N)／X_M←X_M+A_N(X_N−X_M)。
mood-a では未接続（接続は mood-c）。
"""

from __future__ import annotations

import pytest

from familiar_agent.mood_register import (
    MoodPAD,
    SELF_KNOWLEDGE_MI_WEIGHT,
    compute_n_pad,
    nudge_toward,
)


# ── compute_n_pad ───────────────────────────────────────────────────────────
def test_n_pad_empty_is_neutral() -> None:
    # フラット項(0.5)のみ → 中立
    assert compute_n_pad([]) == MoodPAD()


def test_n_pad_single_item_blends_with_the_self_mi() -> None:
    # (w*v + C*self_x)/(w + C)。C=0.5（既定 self_weight）・item weight=2.0。
    # 自己認識 MI の既定は軸ごとの戻り先 (0.10, 0.10, 0.50, 0.50)（案A）。
    #   p  =(2*1.0+0.5*0.10)/2.5=0.82   pn =(2*0.0+0.5*0.10)/2.5=0.02
    #   a  =(2*1.0+0.5*0.50)/2.5=0.90   dom=(2*0.0+0.5*0.50)/2.5=0.10
    n = compute_n_pad([(MoodPAD(1.0, 0.0, 1.0, 0.0), 2.0)])
    assert n == MoodPAD(0.82, 0.02, 0.90, 0.10)


def test_n_pad_heavier_weight_pulls_further_from_flat() -> None:
    light = compute_n_pad([(MoodPAD(1.0, 1.0, 1.0, 1.0), 1.0)])
    heavy = compute_n_pad([(MoodPAD(1.0, 1.0, 1.0, 1.0), 4.0)])
    assert heavy.p > light.p  # 重いほど 0.5 から 1.0 へ寄る


def test_self_knowledge_weight_default_is_half() -> None:
    # 根づき上限 C=2.0 の流用をやめ、支配しない薄い錨（既定 0.5・Config で差替可）
    assert SELF_KNOWLEDGE_MI_WEIGHT == 0.5


# ── nudge_toward ────────────────────────────────────────────────────────────
def test_nudge_full_arousal_moves_to_tone() -> None:
    mood = MoodPAD(0.5, 0.5, 0.3, 0.5)
    n = MoodPAD(0.9, 0.1, 1.0, 0.8)  # A_N=1.0
    out = nudge_toward(mood, n)
    assert out.p == pytest.approx(0.9)
    assert out.pn == pytest.approx(0.1)
    assert out.dom == pytest.approx(0.8)
    assert out.a == pytest.approx(1.0)  # max(0.3, 1.0)


def test_nudge_zero_arousal_keeps_pdom_but_maxes_a() -> None:
    mood = MoodPAD(0.6, 0.4, 0.7, 0.55)
    n = MoodPAD(0.2, 0.9, 0.0, 0.1)  # A_N=0.0
    out = nudge_toward(mood, n)
    assert out.p == pytest.approx(0.6)
    assert out.pn == pytest.approx(0.4)
    assert out.dom == pytest.approx(0.55)
    assert out.a == pytest.approx(0.7)  # max(0.7, 0.0)


def test_nudge_partial_arousal_moves_by_fraction() -> None:
    mood = MoodPAD(0.5, 0.5, 0.5, 0.5)
    n = MoodPAD(1.0, 0.0, 0.4, 1.0)  # A_N=0.4
    out = nudge_toward(mood, n)
    # X_M + A_N*(X_N - X_M) = 0.5 + 0.4*(1.0-0.5) = 0.7
    assert out.p == pytest.approx(0.7)
    assert out.dom == pytest.approx(0.7)
    assert out.pn == pytest.approx(0.5 + 0.4 * (0.0 - 0.5))  # 0.3


def test_nudge_clips_to_range() -> None:
    mood = MoodPAD(0.9, 0.1, 0.9, 0.9)
    n = MoodPAD(1.0, 0.0, 1.0, 1.0)
    out = nudge_toward(mood, n)
    for v in (out.p, out.pn, out.a, out.dom):
        assert 0.0 <= v <= 1.0
