"""Drive 発火 → 自発ターンの選択・ゲート（[D-行動選択]・純関数・Slice 2b）。

発火は固定行動でなく「open 意図」で、具体行動は主LLM が O の傾向＋文脈から選ぶ。
本モジュールは「どの発火軸で・起こしてよいか」だけを決める（DB も loop も持たない）。
"""

from __future__ import annotations

from ..config import DriveConfig
from ..drive_register import AiDrivers
from .drive_dynamics import DriveFiring

# 社会欲求＝在席が要る（不在なら起こさない）。内的欲求は在席に依らない。
SOCIAL_DRIVES: frozenset[str] = frozenset({"bond", "esteem"})

_AXES: tuple[str, ...] = ("seeking", "rest", "bond", "safety", "esteem")


def select_fired_axis(firing: DriveFiring, accumulated: AiDrivers) -> str | None:
    """発火した軸のうち accumulated（放電前）が最大の1軸。無発火なら None。"""
    fired_axes = [ax for ax in _AXES if getattr(firing, ax)]
    if not fired_axes:
        return None
    return max(fired_axes, key=lambda ax: getattr(accumulated, ax))


def inner_voice_for(axis: str, cfg: DriveConfig) -> str:
    """発火軸に対応する内声（Config 文字列・行動非指定）。"""
    return getattr(cfg, f"voice_{axis}")


def drive_gate(
    axis: str,
    *,
    agent_running: bool,
    pending_input: bool,
    quiet: bool,
    presence: float,
) -> bool:
    """この発火軸で自発ターンを起こしてよいか。

    agent 実行中・入力待ち・静穏時間は起こさない。社会欲求は在席ゼロなら起こさない。
    """
    if agent_running or pending_input or quiet:
        return False
    if axis in SOCIAL_DRIVES and presence <= 0.0:
        return False
    return True
