"""Drive 起動源の dynamics（蓄積・気分変調・発火・放電）＝純関数（Slice 1・未接続）。

感情ループ全体像の `T→D`（蓄積）・`M→g_D(M)→D`（気分変調）・`D→FIRE`（発火/放電）を
`AiDrivers`＋`MoodPAD` 上の純関数で実装する。式・値は発火mood §2／課題5 B 由来。
loop（T-tick・自発ターン）への接続と legacy `DesireSystem` の置換は後続スライス。

- 蓄積：`drive_i += rate·mult·learn·g_{D,i}(M)·dt`（clip[0,1]）。
- 変調：`g_{D,i}(M) = logistic(logit(b_i) + Σ_j C_ij·logit(x_j))`（中立 mood で g=b_i）。
- 発火：`drive_i ≥ Θ_fire` で発火、放電 `q` で ~0 へ。I は drive を直接動かさない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import DriveConfig
from ..drive_register import AiDrivers
from ..mood_register import MoodPAD

_EPS = 1e-9


def _logit(x: float) -> float:
    # 0/1 端で ±inf にならないよう内側へ寄せる（PAD は [0,1]・中立0.5→0）。
    x = min(1.0 - _EPS, max(_EPS, x))
    return math.log(x / (1.0 - x))


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _gain(bias: float, coeffs: tuple[float, float, float, float], pad: tuple[float, ...]) -> float:
    z = _logit(bias) + sum(c * _logit(x) for c, x in zip(coeffs, pad))
    return _logistic(z)


def g_d(mood: MoodPAD, cfg: DriveConfig | None = None) -> AiDrivers:
    """各欲求の気分変調ゲイン g_{D,i}(M)（中立 mood で b_i に一致）。"""
    cfg = cfg or DriveConfig()
    pad = (mood.p, mood.pn, mood.a, mood.dom)
    return AiDrivers(
        seeking=_gain(cfg.bias_seeking, cfg.c_seeking, pad),
        rest=_gain(cfg.bias_rest, cfg.c_rest, pad),
        bond=_gain(cfg.bias_bond, cfg.c_bond, pad),
        safety=_gain(cfg.bias_safety, cfg.c_safety, pad),
        esteem=_gain(cfg.bias_esteem, cfg.c_esteem, pad),
    )


def accumulate(
    drives: AiDrivers, mood: MoodPAD, *, dt: float | None = None, cfg: DriveConfig | None = None
) -> AiDrivers:
    """1 tick 蓄積：`drive_i += rate·mult·learn·g_{D,i}(M)·dt`、clip[0,1]。"""
    cfg = cfg or DriveConfig()
    dt = cfg.p_t if dt is None else dt
    g = g_d(mood, cfg)
    step = cfg.rate * cfg.mult * cfg.learn * dt
    return AiDrivers(
        seeking=drives.seeking + step * g.seeking,
        rest=drives.rest + step * g.rest,
        bond=drives.bond + step * g.bond,
        safety=drives.safety + step * g.safety,
        esteem=drives.esteem + step * g.esteem,
    ).clipped()


@dataclass(frozen=True)
class DriveFiring:
    """どの欲求が発火したか（drive_i ≥ Θ_fire）。"""
    seeking: bool = False
    rest: bool = False
    bond: bool = False
    safety: bool = False
    esteem: bool = False

    @property
    def any(self) -> bool:
        return self.seeking or self.rest or self.bond or self.safety or self.esteem


def fired(drives: AiDrivers, cfg: DriveConfig | None = None) -> DriveFiring:
    """各 drive が発火閾値 Θ_fire 以上か。"""
    cfg = cfg or DriveConfig()
    th = cfg.theta_fire
    return DriveFiring(
        seeking=drives.seeking >= th,
        rest=drives.rest >= th,
        bond=drives.bond >= th,
        safety=drives.safety >= th,
        esteem=drives.esteem >= th,
    )


def tick(
    drives: AiDrivers, mood: MoodPAD, *, dt: float, cfg: DriveConfig | None = None
) -> tuple[AiDrivers, DriveFiring]:
    """1 tick 分の dynamics：蓄積 → 発火判定 → 発火なら放電。(新 drives, 発火) を返す。

    ループ接続（Slice 2a）から呼ぶ純関数。DB も loop 制御も持たない（呼び出し側が
    mood/drives を読み、結果を永続化する）。dt は前 tick からの実経過秒。
    """
    cfg = cfg or DriveConfig()
    d = accumulate(drives, mood, dt=dt, cfg=cfg)
    firing = fired(d, cfg)
    if firing.any:
        d = discharge(d, firing, cfg)
    return d, firing


def discharge(drives: AiDrivers, firing: DriveFiring, cfg: DriveConfig | None = None) -> AiDrivers:
    """発火した欲求を放電（q だけ引いて ~0 へ）。発火していない軸は不変。"""
    cfg = cfg or DriveConfig()
    q = cfg.discharge_q

    def d(v: float, f: bool) -> float:
        return max(0.0, min(1.0, v - q)) if f else v

    return AiDrivers(
        seeking=d(drives.seeking, firing.seeking),
        rest=d(drives.rest, firing.rest),
        bond=d(drives.bond, firing.bond),
        safety=d(drives.safety, firing.safety),
        esteem=d(drives.esteem, firing.esteem),
    )
