"""定点ごとの「見えの普通」と、そこからの隔たり。

`知覚在席` §3-4 が定める見え層。定点ごとに DINOv2 埋め込みの「普通」を EMA で持ち、
現フレームとのコサイン距離を $\\widehat{S}_{景色}$ にする。エンティティに現れない見えの変化
（配置のずれ、明るさ）を拾うための層である。

**自己運動では驚かない**（[D-向き]）。定点ごとに別の「普通」を持つので、別の定点へ向いても
その定点の「普通」と比べることになり、向きの違いそのものは驚きにならない。

**育つまでは使わない。** 起動直後は比較対象が無く、最初の1枚をそのまま「普通」にすると、
そのとき写っていたものが基準になる。何回か観測してから距離を出す。
"""

from __future__ import annotations

import math

# `課題5` §I の確定値。1枚の外れ値で「普通」が動きすぎないようにする。
ALPHA_NORM = 0.10
# 「普通」が使えるようになるまでの観測回数。
MIN_OBSERVATIONS = 5


def update_ema(norm: list[float] | None, seen: list[float],
               alpha: float = ALPHA_NORM) -> list[float]:
    """「普通」を1枚ぶん更新する。

    まだ何も無ければ、見えたものをそのまま置く。$\\alpha$ で薄めると原点へ寄ってしまい、
    しばらくのあいだ何と比べても遠い、という状態になる。
    """
    if norm is None:
        return list(seen)
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(norm, seen)]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """$1-\\cos$。値域 $[0,2]$（`課題5` §I）。

    長さを無視する。DINOv2 の出力は正規化されているとは限らず、長さで距離が変わると
    明るさの違いがそのまま驚きになる。片方が零ベクトルなら 0（埋め込みが取れなかった
    ときに例外を投げると、常駐タスクが死ぬ）。
    """
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (na * nb)


def is_ready(observations: int, min_observations: int = MIN_OBSERVATIONS) -> bool:
    """その定点の「普通」が、比較に使える程度に育っているか。"""
    return observations >= min_observations
