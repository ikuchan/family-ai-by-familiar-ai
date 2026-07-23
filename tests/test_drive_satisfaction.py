"""充足放電ルート（案Y）：ターン完了時に軽量LLMが満たされた drive を発火時と同じ全放電。

ゲートは drive 値でなく W/MI・E（PAD 距離）・行動から作る（機構的循環を避ける）。
純関数（ゲート判定・PAD 距離・出力パース・放電適用）をここで検証する。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core.drive_satisfaction import (
    apply_satisfaction,
    pad_distance,
    parse_satisfied_axes,
    satisfaction_gate,
)
from familiar_agent.drive_register import AiDrivers
from familiar_agent.mood_register import MoodPAD


# ── Config：フラグと PAD 距離しきい値 ────────────────────────────────────────

def test_satisfy_flag_default_off():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DRIVE5_SATISFY_LLM", None)
        assert DriveConfig().satisfy_llm is False


def test_satisfy_gate_pad_dist_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DRIVE5_SATISFY_PAD_DIST", None)
        assert DriveConfig().satisfy_gate_pad_dist == 0.2


# ── PAD 距離（L2・上下両方向を拾う） ─────────────────────────────────────────

def test_pad_distance_zero_for_same():
    assert pad_distance(MoodPAD(), MoodPAD()) == pytest.approx(0.0)


def test_pad_distance_captures_downward_move():
    # arousal が下がる（鎮まる・REST 充足方向）も距離として拾う
    d = pad_distance(MoodPAD(a=0.5), MoodPAD(a=0.2))
    assert d == pytest.approx(0.3, abs=1e-9)


# ── ゲート（drive 非依存 OR） ────────────────────────────────────────────────

def test_gate_true_when_memories_nonempty():
    cfg = DriveConfig()
    assert satisfaction_gate(memories_nonempty=True, pad_move=0.0,
                             action_used=False, cfg=cfg) is True


def test_gate_true_when_pad_move_over_threshold():
    cfg = DriveConfig()
    assert satisfaction_gate(memories_nonempty=False, pad_move=0.25,
                             action_used=False, cfg=cfg) is True


def test_gate_true_when_action_used():
    cfg = DriveConfig()
    assert satisfaction_gate(memories_nonempty=False, pad_move=0.0,
                             action_used=True, cfg=cfg) is True


def test_gate_false_when_flat_turn():
    cfg = DriveConfig()
    assert satisfaction_gate(memories_nonempty=False, pad_move=0.1,
                             action_used=False, cfg=cfg) is False


# ── 出力パース（満たされた軸の部分集合） ─────────────────────────────────────

def test_parse_satisfied_axes_from_json():
    assert parse_satisfied_axes('["bond", "rest"]') == frozenset({"bond", "rest"})


def test_parse_satisfied_axes_from_prose():
    assert parse_satisfied_axes("BOND was satisfied") == frozenset({"bond"})


def test_parse_satisfied_axes_ignores_unknown():
    assert parse_satisfied_axes('["bond", "hunger"]') == frozenset({"bond"})


def test_parse_satisfied_axes_empty():
    assert parse_satisfied_axes("none") == frozenset()


# ── 放電適用（発火時と同じ全放電・他軸不変） ─────────────────────────────────

def test_apply_satisfaction_discharges_named_axes():
    out = apply_satisfaction(AiDrivers(bond=0.9, seeking=0.7), frozenset({"bond"}))
    assert out.bond == pytest.approx(0.0, abs=1e-2)   # 全放電
    assert out.seeking == pytest.approx(0.7)          # 他軸は不変


def test_apply_satisfaction_empty_noop():
    src = AiDrivers(bond=0.9)
    out = apply_satisfaction(src, frozenset())
    assert out.bond == pytest.approx(0.9)
