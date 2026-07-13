"""Tests for wiring derived activation into the recall score (Phase 2 P-1).

現行の `importance` を `_derive_activation(activation_a0, activation_n)` へ差し替え、
a 軸を器から想起スコアへ結線する。時間減衰は time_score（t 軸）に一元化し、
importance の日次減衰はスコア経路から外す。ここでは `_compute_final_score` が
(a0, n) から導出 activation を積に用いることを検証する（DB 非依存の単体）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from familiar_agent.tools.memory import _compute_final_score, _derive_activation


_NOW = datetime.now(timezone.utc)


def _score(a0: float, n: int) -> float:
    # 現在時刻・強化なしで time_score をほぼ 1 に寄せ、a 軸の効きを見る
    return _compute_final_score(
        1.0, _NOW, None, 0, a0, n, half_life_days=30.0, floor=0.0
    )


def test_score_uses_derived_activation_not_raw_a0() -> None:
    """n を増やすと score が導出 activation どおり上がる（a 軸が (a0,n) で効く）。"""
    s0 = _score(0.5, 0)
    s3 = _score(0.5, 3)
    assert s3 > s0
    # 比は導出 activation の比に一致（time_score は共通で相殺）
    ratio = _derive_activation(0.5, 3) / _derive_activation(0.5, 0)
    assert abs((s3 / s0) - ratio) < 1e-6


def test_score_n_zero_equals_a0_factor() -> None:
    """n=0 では導出 activation ＝ a0。score は cosine×time_score×a0 に一致。"""
    a0 = 0.7
    s = _score(a0, 0)
    # 同条件で a0 を 1.0 にした基準との比が a0 に一致（time_score 相殺）
    base = _score(1.0, 0)
    assert abs((s / base) - a0) < 1e-6
