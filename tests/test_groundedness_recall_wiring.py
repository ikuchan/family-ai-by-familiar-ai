"""Tests for wiring derived 根づき into the recall score (Phase 2 P-1).

現行の `importance` を `_derive_groundedness(groundedness_g0, groundedness_n)` へ差し替え、
a 軸を器から想起スコアへ結線する。時間減衰は time_score（t 軸）に一元化し、
importance の日次減衰はスコア経路から外す。ここでは `_compute_final_score` が
(a0, n) から導出した 根づき を a 軸に用いることを検証する（DB 非依存の単体）。
スライス3 で合成がハイブリッドになったので、a は積の因子ではなく加算部 M の
一項（係数 w_g=1.5）として効く。期待値をその式へ合わせてある。
"""

from __future__ import annotations

from datetime import datetime, timezone

from familiar_agent.tools.memory import _compute_final_score, _derive_groundedness


def _score(a0: float, n: int) -> float:
    # 現在時刻・強化なしで time_score をほぼ 1 に寄せ、a 軸の効きを見る。
    # now はモジュール読み込み時でなく呼び出し時に取る（読み込み時に固定すると、
    # 全体テストのように実行まで時間が空いたとき t が減衰して期待値とずれる）。
    now = datetime.now(timezone.utc)
    return _compute_final_score(
        1.0, now, None, 0, a0, n, half_life_days=30.0, floor=0.0
    )


def _expected(a0: float, n: int) -> float:
    """スライス3 のハイブリッド：cosine=1・t≈1・mood 無しなので (t + 1.5a)/2.5。"""
    return (1.0 + 1.5 * _derive_groundedness(a0, n)) / 2.5


def test_score_uses_derived_groundedness_not_raw_a0() -> None:
    """n を増やすと score が導出した根づき どおり上がる（a 軸が (a0,n) で効く）。"""
    s0 = _score(0.5, 0)
    s3 = _score(0.5, 3)
    assert s3 > s0
    # 加算部の一項として効くので、上がり幅は導出した根づき の差の w_g/Σw 倍
    delta = 1.5 * (_derive_groundedness(0.5, 3) - _derive_groundedness(0.5, 0)) / 2.5
    assert abs((s3 - s0) - delta) < 1e-6


def test_score_n_zero_equals_a0_factor() -> None:
    """n=0 では導出した根づき ＝ a0。score は (t + 1.5·a0)/2.5 に一致。"""
    a0 = 0.7
    assert abs(_score(a0, 0) - _expected(a0, 0)) < 1e-6
    assert abs(_score(1.0, 0) - _expected(1.0, 0)) < 1e-6
