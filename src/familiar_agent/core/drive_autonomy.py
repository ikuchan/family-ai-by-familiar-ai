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


# drive5 スナップショットの表示順（発火mood の値表順）と表示名。
_SNAPSHOT_ORDER: tuple[str, ...] = ("seeking", "safety", "bond", "esteem", "rest")


def qualitative_level(value: float, cfg: DriveConfig) -> str:
    """drive 値を定性ラベルへ（低<mid / 中 / 高≥high・生値は出さない）。"""
    if value >= cfg.drive_level_high:
        return "高"
    if value >= cfg.drive_level_mid:
        return "中"
    return "低"


def drive_snapshot(drives: AiDrivers, cfg: DriveConfig) -> str:
    """5軸の状態を定性ラベルで1行に（発火軸以外の欲求バランスもターンへ渡す）。"""
    parts = [
        f"{ax.upper()} {qualitative_level(getattr(drives, ax), cfg)}"
        for ax in _SNAPSHOT_ORDER
    ]
    return " / ".join(parts)


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
