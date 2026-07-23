"""充足放電ルート（案Y・純関数）：ターン完了時の軽量LLM充足判定で drive を全放電。

感情ループ正本は「I は D を直接動かさない」（発火mood §2.5）が、その充足経路（mood 媒介）
は現状ほぼ効いていない。例外ルートとして、ターン結果を軽量LLMが読み、満たされた drive を
発火時と同じ全放電で沈静化する。ゲートは drive 値を使わず（鎮静対象をその値でゲートする循環を
避ける）、W/MI・E（PAD 距離・上下両方向）・行動から作る。本モジュールは判定・パース・放電適用の
純関数だけを持つ（LLM 呼び出しと DB は呼び出し側）。
"""

from __future__ import annotations

import math

from ..config import DriveConfig
from ..drive_register import AiDrivers
from ..mood_register import MoodPAD
from .drive_dynamics import DriveFiring, discharge

_AXES: tuple[str, ...] = ("seeking", "rest", "bond", "safety", "esteem")


def pad_distance(a: MoodPAD, b: MoodPAD) -> float:
    """2つの mood の PAD 距離（L2・4軸）。上下どちらの動きも正の距離で拾う。"""
    return math.sqrt(
        (a.p - b.p) ** 2 + (a.pn - b.pn) ** 2 + (a.a - b.a) ** 2 + (a.dom - b.dom) ** 2
    )


def satisfaction_gate(
    *,
    memories_nonempty: bool,
    pad_move: float,
    action_used: bool,
    cfg: DriveConfig,
) -> bool:
    """充足判定LLM を回すか（drive 値は使わない・実質的ターンかを W/MI・E・行動で判定）。"""
    return memories_nonempty or pad_move >= cfg.satisfy_gate_pad_dist or action_used


def parse_satisfied_axes(text: str) -> frozenset[str]:
    """軽量LLMの出力から満たされた軸の部分集合を取り出す（未知語は無視）。"""
    lower = text.lower()
    return frozenset(ax for ax in _AXES if ax in lower)


def apply_satisfaction(
    drives: AiDrivers, axes: "frozenset[str] | set[str]", cfg: DriveConfig | None = None
) -> AiDrivers:
    """満たされた軸を発火時と同じ全放電で沈静化する（他軸は不変）。"""
    if not axes:
        return drives
    firing = DriveFiring(**{ax: (ax in axes) for ax in _AXES})
    return discharge(drives, firing, cfg)
