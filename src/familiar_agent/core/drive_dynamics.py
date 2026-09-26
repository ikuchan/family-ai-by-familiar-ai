"""Drive 起動源の dynamics（蓄積・気分変調・発火・放電）＝純関数。

感情ループ全体像の `T→D`（蓄積）・`M→g_D(M)→D`（気分変調）・`D→FIRE`（発火/放電）を
`AiDrivers`＋`MoodPAD` 上の純関数で実装する。式・値は発火mood §2／課題5 B 由来。
**loop へは接続済み**（2026-09-10 確認）——`loop/tonic.py` が `accumulate` → `fired` →
`discharge` を時間で回して永続化する。旧 15 欲求の系は環-d で撤去した
（2026-09-15）。自律の駆動源は T である。

- 蓄積：`drive_i += rate·mult·learn·g_{D,i}(M)·dt`（clip[0,1]）。
- 変調：`g_{D,i}(M) = logistic(logit(b_i) + Σ_j C_ij·(logit(x_j) − logit(r_j)))`
  （`r` は軸ごとの平静＝`REST_PAD`。平静 mood で g=b_i）。
- 発火：`drive_i ≥ Θ_fire` で発火、放電 `q` で ~0 へ。I は drive を直接動かさない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..config import DriveConfig
from ..drive_register import AiDrivers
from ..mood_register import REST_PAD, MoodPAD
from .solitude import Solitude

#: 誰も映っていないとき、BOND・ESTEEM が減る速さ＝溜まる速さの 1/SOCIAL_FADE（出-as・本人の決定 1/3）。
SOCIAL_FADE = 3.0

_EPS = 1e-9


def _logit(x: float) -> float:
    # 0/1 端で ±inf にならないよう内側へ寄せる（PAD は [0,1]・中立0.5→0）。
    x = min(1.0 - _EPS, max(_EPS, x))
    return math.log(x / (1.0 - x))


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _gain(
    bias: float,
    coeffs: tuple[float, float, float, float],
    pad: tuple[float, ...],
    rest: tuple[float, ...],
) -> float:
    z = _logit(bias) + sum(c * (_logit(x) - _logit(r)) for c, x, r in zip(coeffs, pad, rest))
    return _logistic(z)


def g_d(mood: MoodPAD, cfg: DriveConfig | None = None, *, rest: MoodPAD = REST_PAD) -> AiDrivers:
    """各欲求の気分変調ゲイン g_{D,i}(M)（平静 mood で b_i に一致）。

    各項は**平静からのずれ**である。ロジットを絶対値で足すと項が消えるのは 0.5 の
    ときだけで、案A で快と不快の平静が 0.10 へ動いたあとは、何も起きていなくても
    蓄積速度がずれてしまう。差にしておけば、平静をどこに置いても
    「平静では g = b_i」（`感情ループ全体像`）が成り立つ。
    """
    cfg = cfg or DriveConfig()
    pad = (mood.p, mood.pn, mood.a, mood.dom)
    rest_t = (rest.p, rest.pn, rest.a, rest.dom)
    return AiDrivers(
        seeking=_gain(cfg.bias_seeking, cfg.c_seeking, pad, rest_t),
        rest=_gain(cfg.bias_rest, cfg.c_rest, pad, rest_t),
        bond=_gain(cfg.bias_bond, cfg.c_bond, pad, rest_t),
        safety=_gain(cfg.bias_safety, cfg.c_safety, pad, rest_t),
        esteem=_gain(cfg.bias_esteem, cfg.c_esteem, pad, rest_t),
    )


def accumulate(
    drives: AiDrivers,
    mood: MoodPAD,
    *,
    dt: float | None = None,
    cfg: DriveConfig | None = None,
    solitude: Solitude | None = None,
    someone_visible: bool = True,
) -> AiDrivers:
    """1 tick 蓄積：`drive_i += rate·mult_i(t)·learn·2^(−n_i)·g_{D,i}(M)·dt`、clip[0,1]。

    倍率は軸ごとに違う（`cfg.mult_for`）。深夜は探索ほかを抑える一方、REST は逆に
    募らせるためである。時刻の判定は T が済ませてあり、ここは軸名で引くだけにする。

    `solitude`（ひとりの回数・情-d）は、人と会話しないかぎり SEEKING・SAFETY・BOND の
    間隔を倍々に伸ばす。None なら 1 倍（従来どおり）。

    `someone_visible`（カメラに人が映っているか・出-as §2.1・2026-09-26）：人に話しかけたくなる BOND と
    ESTEEM は、**映っているあいだだけ溜まり、映っていなければ溜まる速さの 1/3 で減る**（本人の決定）。
    ほかの 3 軸は映っているかに関わらない。
    """
    cfg = cfg or DriveConfig()
    dt = cfg.p_t if dt is None else dt
    g = g_d(mood, cfg)
    base = cfg.rate * cfg.learn * dt
    lonely = solitude or Solitude()

    def _step(axis: str) -> float:
        return base * cfg.mult_for(axis) * lonely.factor(axis)

    def _social(axis: str, gain: float) -> float:
        grow = _step(axis) * gain
        return grow if someone_visible else -grow / SOCIAL_FADE

    return AiDrivers(
        seeking=drives.seeking + _step("seeking") * g.seeking,
        rest=drives.rest + _step("rest") * g.rest,
        bond=drives.bond + _social("bond", g.bond),
        safety=drives.safety + _step("safety") * g.safety,
        esteem=drives.esteem + _social("esteem", g.esteem),
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


def nudge(drives: AiDrivers, axis: str, amount: float, cfg: DriveConfig | None = None) -> AiDrivers:
    """出来事で 1 軸を押し上げる（蓄積への加算・他軸は触らない）。上限は発火閾値。

    押し上げは蓄積の一部で、**発火は通常の tick（`fired`）が決める**。出来事から直接
    発火させると、間隔の伸び（情-d）や静穏時間の抑えを飛び越えてしまう。
    使い手：声がしたが応じられないとき SEEKING へ（案ア・2026-09-17）。
    """
    cfg = cfg or DriveConfig()
    if axis not in ("seeking", "rest", "bond", "safety", "esteem"):
        return drives
    value = min(cfg.theta_fire, max(0.0, getattr(drives, axis) + max(0.0, amount)))
    return replace(drives, **{axis: value})


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
