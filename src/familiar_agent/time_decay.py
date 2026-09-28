"""Pure time-decay value object. No DB, no env, no settings — calculation only.

Used by memory recall (Issue B).

Decay formula: score = max(floor, exp(-elapsed / tau))
               tau   = effective_half_life / ln(2)

Reinforcement A (durability): reinforce_count += 1 → effective half-life doubles.
Reinforcement B (freshness):  origin_epoch reset → elapsed restarts from zero.

Memory uses both A and B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DecayState:
    origin_epoch: float
    half_life_seconds: float
    floor: float = 0.0
    reinforce_count: int = 0

    def score(self, reference_epoch: float) -> float:
        """基準時刻からの**隔たり**で減衰させる（両側・絶対値）。

        引数は「いま」ではなく**基準時刻**である。調停が人の言葉から基準を動かせるので
        （「去年の夏の話」）、記録が基準より後にあることもある。以前は
        `max(0.0, now - origin)` と負を切り捨てており、基準を過去へ動かすと**それ以降の
        記録が全部 t=1** になっていた。
        """
        if self.half_life_seconds <= 0:
            return self.floor
        effective = self.half_life_seconds * (2 ** max(0, self.reinforce_count))
        tau = effective / math.log(2)
        distance = abs(reference_epoch - self.origin_epoch)
        return max(self.floor, math.exp(-distance / tau))
