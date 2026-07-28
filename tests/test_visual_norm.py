"""定点ごとの「見えの普通」と、そこからの隔たり。

`知覚在席` §3-4：定点ごとに見えの「普通」を EMA で持ち、現フレームとの距離を
$\\widehat{S}_{景色}$ にする。`課題5` §I が定める式は次のとおり。

- 距離＝DINOv2 埋め込みの定点別「普通(EMA)」との**コサイン距離** $1-\\cos$（値域 $[0,2]$）
- EMA 係数 $\\alpha_{norm} = 0.10$

**育つまでは距離を出さない。** 起動直後は比較対象が無く、最初の1枚をそのまま「普通」に
すると、そのとき写っていたものが基準になる。何回か観測してから使う（既定5回）。

自己運動では驚かない（[D-向き]）。定点ごとに別の「普通」を持つので、別の定点へ向いても
その定点の「普通」と比べることになり、向きの違いが驚きにならない。
"""

from __future__ import annotations

import math

from familiar_agent.visual_norm import cosine_distance, is_ready, update_ema


def _unit(values: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in values))
    return [v / n for v in values]


# --- 「普通」の更新 -------------------------------------------------------


def test_the_first_observation_becomes_the_norm_as_is():
    # 比較対象が無いので、最初はそのまま置く（α で薄めると原点へ寄ってしまう）。
    assert update_ema(None, [1.0, 0.0], alpha=0.1) == [1.0, 0.0]


def test_the_norm_moves_towards_what_is_seen():
    got = update_ema([1.0, 0.0], [0.0, 1.0], alpha=0.1)
    assert abs(got[0] - 0.9) < 1e-9
    assert abs(got[1] - 0.1) < 1e-9


def test_a_small_alpha_keeps_the_norm_stable():
    # α=0.10（`課題5` §I）。1枚の外れ値で「普通」が動きすぎない。
    got = update_ema([1.0, 0.0], [0.0, 1.0], alpha=0.10)
    assert got[0] > got[1]


def test_seeing_the_same_thing_leaves_the_norm_where_it_is():
    got = update_ema([0.6, 0.8], [0.6, 0.8], alpha=0.1)
    assert all(abs(g - e) < 1e-9 for g, e in zip(got, [0.6, 0.8]))


# --- 隔たり ---------------------------------------------------------------


def test_the_same_view_is_zero_distance():
    v = _unit([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) < 1e-9


def test_an_orthogonal_view_is_distance_one():
    assert abs(cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9


def test_the_opposite_view_is_distance_two():
    # 値域は [0,2]（`課題5` §I）。実用は [0,1] 寄り。
    assert abs(cosine_distance([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-9


def test_distance_ignores_the_length_of_the_vectors():
    # DINOv2 の出力は正規化されているとは限らない。長さで距離が変わると、明るさの違いが
    # そのまま驚きになる。
    assert cosine_distance([1.0, 0.0], [5.0, 0.0]) < 1e-9


def test_a_zero_vector_yields_no_distance_rather_than_an_error():
    # 埋め込みが取れなかったときに例外を投げると、常駐タスクが死ぬ。
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- 育つまで -------------------------------------------------------------


def test_the_norm_is_not_usable_before_it_has_grown():
    assert is_ready(0, min_observations=5) is False
    assert is_ready(4, min_observations=5) is False


def test_the_norm_becomes_usable_at_the_threshold():
    assert is_ready(5, min_observations=5) is True
    assert is_ready(50, min_observations=5) is True
