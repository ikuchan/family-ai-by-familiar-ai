"""Tests for the hybrid recall score (Phase 2 スライス3・e 軸のスコア接続).

合成を純積から課題5 v0.24 D 節のハイブリッドへ切り替える。

    score = r^{w_r} × M,  M = (w_t·t + w_e·e + w_g·a) / (w_t + w_e + w_g)

在席者ゼロ（p 軸は知覚待ちで項ごと外す）の基底プロファイルは
(w_r, w_t, w_e, w_g) = (1, 1, 1, 1.5) すなわち score = r·(t + e + 1.5a)/3.5。
DB 非依存の単体。
"""

from __future__ import annotations

from datetime import datetime, timezone

from familiar_agent.tools.memory import (
    _compute_final_score,
    _derive_groundedness,
    _emotion_match,
)


_NEUTRAL = (0.5, 0.5, 0.5, 0.5)


def _now() -> datetime:
    """呼び出し時刻。モジュール読み込み時に固定すると、全体テストのように
    実行までに時間が空いたとき t が 1.0 から減衰して手計算とずれる。"""
    return datetime.now(timezone.utc)


def _score(
    cosine: float = 1.0,
    *,
    a0: float = 1.0,
    n: int = 0,
    obs_pad=_NEUTRAL,
    mood_pad=_NEUTRAL,
    **kw,
) -> float:
    """last_recalled_at を now にして t を 1.0 に固定し、他軸の効きを見る。"""
    return _compute_final_score(
        cosine,
        _now(),
        _now(),
        0,
        a0,
        n,
        obs_pad=obs_pad,
        mood_pad=mood_pad,
        half_life_days=3.0,
        floor=0.001,
        **kw,
    )


def test_matches_hand_computed_base_profile() -> None:
    """基底 (1,1,1,1.5) で score = r·(t + e + 1.5a)/3.5 に一致する。"""
    cosine, a0, n = 0.8, 1.0, 0
    obs = (0.9, 0.2, 0.7, 0.6)
    a = _derive_groundedness(a0, n)
    e = _emotion_match(obs, _NEUTRAL, sigma=1.0)
    expected = cosine * (1.0 + e + 1.5 * a) / 3.5
    got = _score(cosine, a0=a0, n=n, obs_pad=obs)
    assert abs(got - expected) < 1e-6


def test_emotion_axis_affects_score() -> None:
    """観測 PAD が mood から遠いほどスコアが下がる（e が効いている）。"""
    near = _score(obs_pad=(0.55, 0.5, 0.5, 0.5))
    far = _score(obs_pad=(0.95, 0.5, 0.5, 0.5))
    assert near > far


def test_emotion_is_not_a_veto() -> None:
    """e がほぼ 0 でも、a が高ければスコアは 0 にならない。

    これが純積からハイブリッドへ替える理由そのもの。乗算で e を足すと
    感情の遠い記憶が一律に沈む（拒否権になる）ため、加算部の一項にする。
    """
    far = (0.999, 0.999, 0.999, 0.999)
    e = _emotion_match(far, _NEUTRAL, sigma=1.0)
    assert e < 1e-6  # 前提：この PAD で e は実質ゼロ
    got = _score(a0=1.5, n=0, obs_pad=far)
    assert got > 0.3


def test_mood_none_drops_the_emotion_term() -> None:
    """mood が読めないときは e 項を分子分母から外す（中立0.5で埋めない）。"""
    a0, n = 1.2, 2
    a = _derive_groundedness(a0, n)
    expected = 1.0 * (1.0 + 1.5 * a) / 2.5
    got = _score(a0=a0, n=n, obs_pad=(0.9, 0.1, 0.1, 0.9), mood_pad=None)
    assert abs(got - expected) < 1e-6


def test_w_r_zero_disables_relevance() -> None:
    """w_r=0 で関連ゲートが無効化される（r^0 = 1）。"""
    lo = _score(0.2, w_r=0.0)
    hi = _score(0.9, w_r=0.0)
    # 2回の呼び出しの間に now が進むぶん t が僅かに減るので、厳密一致ではなく
    # コサインの差(0.7)に比べて無視できることを見る。
    assert abs(lo - hi) < 1e-6


def test_additive_part_all_zero_gives_m_one() -> None:
    """加算部の重みが全0なら M=1（score は r だけになる）。"""
    got = _score(0.6, w_t=0.0, w_e=0.0, w_g=0.0)
    assert abs(got - 0.6) < 1e-9
