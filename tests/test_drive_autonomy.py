"""Drive Slice 2b：発火→自発ターンの純関数と Config（[D-行動選択]・排他フラグ）。

- DRIVE5_AUTONOMOUS：新Drive発火で自発ターンを起こすか（既定 off＝legacy と完全排他）。
- 内声は Config 文字列（env 上書き可）で持ち、行動は指定しない（主LLM が O と文脈から選ぶ）。
- select_fired_axis：同時発火時は accumulated（放電前）最大の1軸を選ぶ。
- drive_gate：agent実行中/入力待ち/静穏で棄却。社会欲求(BOND/ESTEEM)は在席ゼロで棄却。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent.config import DriveConfig
from familiar_agent.core.drive_autonomy import (
    SOCIAL_DRIVES,
    drive_gate,
    inner_voice_for,
    select_fired_axis,
)
from familiar_agent.core.drive_dynamics import DriveFiring
from familiar_agent.drive_register import AiDrivers


# ── Config：排他フラグと内声 ─────────────────────────────────────────────────

def test_autonomous_flag_default_off():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DRIVE5_AUTONOMOUS", None)
        assert DriveConfig().autonomous is False


def test_autonomous_flag_env_on():
    with patch.dict(os.environ, {"DRIVE5_AUTONOMOUS": "1"}, clear=False):
        assert DriveConfig().autonomous is True


def test_inner_voice_present_for_all_axes():
    cfg = DriveConfig()
    for axis in ("seeking", "rest", "bond", "safety", "esteem"):
        assert inner_voice_for(axis, cfg).strip()  # 空でない


# ── select_fired_axis ────────────────────────────────────────────────────────

def test_select_fired_axis_none_when_no_firing():
    assert select_fired_axis(DriveFiring(), AiDrivers()) is None


def test_select_fired_axis_single():
    f = DriveFiring(seeking=True)
    assert select_fired_axis(f, AiDrivers(seeking=1.0)) == "seeking"


def test_select_fired_axis_picks_highest_accumulated():
    # bond と safety が同時発火 → accumulated（放電前）が大きい方
    f = DriveFiring(bond=True, safety=True)
    acc = AiDrivers(bond=1.02, safety=1.20)
    assert select_fired_axis(f, acc) == "safety"


# ── drive_gate ───────────────────────────────────────────────────────────────

def test_gate_blocks_when_agent_running():
    assert drive_gate("seeking", agent_running=True, pending_input=False,
                      quiet=False, presence=0.0) is False


def test_gate_blocks_when_pending_input():
    assert drive_gate("seeking", agent_running=False, pending_input=True,
                      quiet=False, presence=0.0) is False


def test_gate_blocks_when_quiet():
    assert drive_gate("seeking", agent_running=False, pending_input=False,
                      quiet=True, presence=0.0) is False


def test_gate_internal_axis_passes_without_presence():
    assert drive_gate("seeking", agent_running=False, pending_input=False,
                      quiet=False, presence=0.0) is True


def test_gate_social_axis_blocked_without_presence():
    assert drive_gate("bond", agent_running=False, pending_input=False,
                      quiet=False, presence=0.0) is False


def test_gate_social_axis_passes_with_presence():
    assert drive_gate("bond", agent_running=False, pending_input=False,
                      quiet=False, presence=1.0) is True


def test_social_drives_are_bond_and_esteem():
    assert SOCIAL_DRIVES == frozenset({"bond", "esteem"})
