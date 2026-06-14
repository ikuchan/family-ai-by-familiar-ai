"""Pure time-decay value object. No DB, no env, no settings — calculation only.

Used by memory recall (Issue B) and, later, pending_speech intent freshness (Issue D).

Decay formula: score = max(floor, exp(-elapsed / tau))
               tau   = effective_half_life / ln(2)

Reinforcement A (durability): reinforce_count += 1 → effective half-life doubles.
Reinforcement B (freshness):  origin_epoch reset → elapsed restarts from zero.

Memory uses both A and B. pending_speech (Issue D) will use B only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DecayState:
    origin_epoch: float
    half_life_seconds: float
    floor: float = 0.0
    reinforce_count: int = 0

    def score(self, now_epoch: float) -> float:
        if self.half_life_seconds <= 0:
            return self.floor
        effective = self.half_life_seconds * (2 ** max(0, self.reinforce_count))
        tau = effective / math.log(2)
        elapsed = max(0.0, now_epoch - self.origin_epoch)
        return max(self.floor, math.exp(-elapsed / tau))

    def reinforced_durability(self) -> DecayState:
        """強化A: reinforce_count+1 → 実効半減期が2倍になる。"""
        return replace(self, reinforce_count=self.reinforce_count + 1)

    def reinforced_freshness(self, now_epoch: float) -> DecayState:
        """強化B: origin_epoch をリセット → 経過時間がゼロに戻る。"""
        return replace(self, origin_epoch=now_epoch)
