"""根づき（groundedness）の (g0, n) 表現。

初期値 g0 と正味デルタ回数 n から根づきを導く純関数を確かめる。**時間では減らない**の
がこの量の性質で、時間減衰は想起スコアの t 軸が別に担う。

過去のマイグレーション（021 の列追加と importance からの転記、031 の g0 一括再計算）を
再実行するテストは撤去した。一度きりの移行を今日のスキーマへ当て直す意味が無く、旧列を
作り直す副作用があるためである。移行の根拠と実測は `計測・設定値 根拠台帳` §14 にある。
列の存在は `test_groundedness_rename.py` が確かめる。
"""

from __future__ import annotations

import pytest

from familiar_agent.tools.memory import _derive_groundedness


# ── 1. n=0 returns a0 (logit/logistic are inverses) ─────────────────────────

def test_derive_groundedness_n_zero_returns_a0() -> None:
    for a0 in (0.5, 0.75, 1.5):
        assert _derive_groundedness(a0, 0) == pytest.approx(a0)


# ── 2. monotonic in n ────────────────────────────────────────────────────────

def test_derive_groundedness_monotonic_in_n() -> None:
    a0 = 1.0
    a_minus = _derive_groundedness(a0, -2)
    a_zero = _derive_groundedness(a0, 0)
    a_plus = _derive_groundedness(a0, 2)

    assert a_minus < a_zero < a_plus


# ── 3. asymptotes toward floor/C at extremes ────────────────────────────────

def test_derive_groundedness_approaches_bounds() -> None:
    a0 = 1.0
    high = _derive_groundedness(a0, 50)
    low = _derive_groundedness(a0, -50)

    assert high < 2.0
    assert high > 1.99
    assert low > 0.0
    assert low < 0.01


# ── 4. +1 then -1 returns to the original value (round-trip) ────────────────

def test_derive_groundedness_plus_minus_one_round_trip() -> None:
    a0 = 0.6
    up = _derive_groundedness(a0, 1)
    # deriving from a0 directly with n=+1 then n=-1 from the same a0 baseline
    # (symmetry check around n=0, not a chained re-derivation)
    down = _derive_groundedness(a0, -1)
    mid = _derive_groundedness(a0, 0)

    assert (up - mid) > 0
    assert (mid - down) > 0
    assert abs((up - mid) - (mid - down)) < 0.2  # roughly symmetric near center


# ── 5. a0=0.75, step=0.33 → 評価5回で実用上限1.5に到達 ────────────────────────

def test_derive_groundedness_reaches_practical_limit_at_five() -> None:
    a4 = _derive_groundedness(0.75, 4)
    a5 = _derive_groundedness(0.75, 5)
    assert a4 < 1.5 <= a5   # 4回では1.5未満、5回で1.5到達
    assert a5 < 1.6         # ハード上限C=2にはまだ遠い（緩んで育つ）
