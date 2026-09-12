"""ひとりの回数（solitude）——人と会話しないかぎり欲求の発火間隔を倍々に伸ばす（情-d）。

中立の気分で SEEKING は 5 分ごとに必ず発火し、実機では 1 時間に 16 回起きて費用の大半に
なっていた（2026-09-12）。人と会話したかどうかは蓄積の式に入っていなかった。

**軸ごとの整数 n_i を持ち、蓄積の倍率に 2^(−n_i) を掛ける。** 発火で n_i += 1、人の発話で
全部 0 へ戻る（会話で基準の間隔へ戻る）。REST・ESTEEM は持たない（REST は夜に募らせる別の
規則があり、ESTEEM は 12 時間に 1 回で十分に遅い）。値は `課題5` B 章。時計は持たない
（更新は T の tick が行う・`loop/tonic.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: 倍々に伸ばす軸。REST・ESTEEM は含めない。
AXES = ("seeking", "safety", "bond")

#: 頭打ち。2^20 ≈ 100 万倍で、基準 5 分なら約 10 年——実質「会話まで起きない」。
MAX_COUNT = 20


@dataclass(frozen=True)
class Solitude:
    """ひとりの回数（軸ごと）と、最後に会話を見た時刻（epoch 秒・None は未記録）。"""

    seeking: int = 0
    safety: int = 0
    bond: int = 0
    reset_at: float | None = None

    def factor(self, axis: str) -> float:
        """その軸の蓄積に掛ける倍率 2^(−n)。持たない軸は 1.0。"""
        if axis not in AXES:
            return 1.0
        return 0.5 ** int(getattr(self, axis))

    def fired(self, axis: str) -> "Solitude":
        """その軸が発火した。数える（持たない軸はそのまま）。"""
        if axis not in AXES:
            return self
        return replace(self, **{axis: min(MAX_COUNT, int(getattr(self, axis)) + 1)})

    def reset(self, *, at: float) -> "Solitude":
        """人と会話した。全部 0 へ戻し、その時刻を控える。"""
        return Solitude(reset_at=at)

    def to_json_dict(self) -> dict:
        return {
            "seeking": self.seeking,
            "safety": self.safety,
            "bond": self.bond,
            "reset_at": self.reset_at,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "Solitude":
        return cls(
            seeking=int(d.get("seeking", 0)),
            safety=int(d.get("safety", 0)),
            bond=int(d.get("bond", 0)),
            reset_at=(None if d.get("reset_at") is None else float(d["reset_at"])),
        )


def next_interval_minutes(axis: str, solitude: Solitude, cfg) -> float:
    """中立の気分で、次にその軸が発火するまでの見込み（分）。ログ用。

    基準は `rate·mult·learn·bias` で 0 から Θ_fire まで溜まる時間。ひとりの回数で 2^n 倍。
    """
    bias = float(getattr(cfg, f"bias_{axis}"))
    per_second = cfg.rate * cfg.mult_for(axis) * cfg.learn * bias * solitude.factor(axis)
    if per_second <= 0:
        return float("inf")
    return cfg.theta_fire / per_second / 60.0
