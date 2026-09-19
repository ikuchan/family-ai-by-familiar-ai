"""鳴る音の山谷（2026-09-19）。純関数。

30 秒鳴らし、8 秒で小さく（倍率 0.1）、25 秒を過ぎたら元の大きさ（1.5）に（本人が聴いて決めた・2026-09-19）。
音は 1 秒の wav の繰り返しで、繰り返すたびにその時点の倍率で再生する（`DIF._ring`）。小さい区間の倍率は
比率でなく**値**で持つ（`RING_SOFT_GAIN`）。
"""

from __future__ import annotations


def gain_at(
    elapsed: float, base: float, *, soft_after: float, soft_until: float, soft_gain: float
) -> float:
    """`elapsed` 秒の時点の倍率。`soft_after` 以上 `soft_until` 未満は `soft_gain`、それ以外は `base`。"""
    if soft_after <= elapsed < soft_until:
        return soft_gain
    return base
