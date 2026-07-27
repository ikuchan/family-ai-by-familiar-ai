"""Tests for DecayState pure value object (time_decay.py)."""
from __future__ import annotations

import dataclasses
import math

import pytest

from familiar_agent.time_decay import DecayState


def test_score_at_origin_is_one():
    """経過0なら係数1.0（floorより上）。"""
    s = DecayState(origin_epoch=1000.0, half_life_seconds=86400.0, floor=0.25)
    assert abs(s.score(1000.0) - 1.0) < 1e-9


def test_score_at_half_life_is_half():
    """1半減期経過で約0.5。"""
    hl = 86400.0
    s = DecayState(origin_epoch=0.0, half_life_seconds=hl, floor=0.0)
    assert abs(s.score(hl) - 0.5) < 1e-6


def test_score_respects_floor():
    """十分時間が経っても floor を下回らない。"""
    s = DecayState(origin_epoch=0.0, half_life_seconds=10.0, floor=0.25)
    assert s.score(1e9) == 0.25


def test_score_is_symmetric_around_the_reference():
    """基準より**後**の記録も、隔たりに応じて減衰する（両側・絶対値）。

    以前は `max(0, now - origin)` と負を切り捨てており、1.0 で頭打ちだった。基準時刻を
    調停が過去へ動かせるようになったので（「去年の夏の話」）、それより後の記録が全部
    1.0 になってしまう。隔たりは絶対値で測る。
    """
    ref = 1000.0
    before = DecayState(origin_epoch=ref - 500.0, half_life_seconds=86400.0, floor=0.0)
    after  = DecayState(origin_epoch=ref + 500.0, half_life_seconds=86400.0, floor=0.0)
    assert before.score(ref) == pytest.approx(after.score(ref))
    assert after.score(ref) < 1.0
    # 基準そのものは 1.0。
    assert DecayState(origin_epoch=ref, half_life_seconds=86400.0, floor=0.0).score(ref) == 1.0


def test_reinforced_durability_extends_half_life():
    """強化A: reinforce_count+1 で実効半減期が2倍 → 同経過でスコアが上がる。"""
    base = DecayState(origin_epoch=0.0, half_life_seconds=86400.0, floor=0.0)
    strong = base.reinforced_durability()
    t = 86400.0
    assert strong.score(t) > base.score(t)
    assert strong.reinforce_count == 1
    assert base.reinforce_count == 0  # 元は不変


def test_reinforced_durability_doubles_effective_half_life():
    """強化A後の半減期が2倍になることを数値で確認。"""
    hl = 86400.0
    base = DecayState(origin_epoch=0.0, half_life_seconds=hl, floor=0.0)
    strong = base.reinforced_durability()
    # strong の1半減期 = hl (base の半減期) → score ≈ 1/sqrt(2) ≈ 0.707
    assert abs(strong.score(hl) - 1 / math.sqrt(2)) < 1e-6


def test_reinforced_freshness_resets_origin():
    """強化B: 起点が現在にリセット → スコアが1.0付近に戻る。"""
    s = DecayState(origin_epoch=0.0, half_life_seconds=100.0, floor=0.0)
    refreshed = s.reinforced_freshness(1000.0)
    assert abs(refreshed.score(1000.0) - 1.0) < 1e-9
    assert refreshed.origin_epoch == 1000.0
    assert s.origin_epoch == 0.0  # 元は不変


def test_reinforced_freshness_preserves_other_fields():
    """強化Bは origin_epoch だけ変え、他フィールドはそのまま。"""
    s = DecayState(origin_epoch=0.0, half_life_seconds=500.0, floor=0.1, reinforce_count=2)
    refreshed = s.reinforced_freshness(999.0)
    assert refreshed.half_life_seconds == 500.0
    assert refreshed.floor == 0.1
    assert refreshed.reinforce_count == 2


def test_decaystate_is_frozen():
    """不変であること。"""
    s = DecayState(origin_epoch=0.0, half_life_seconds=100.0, floor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.origin_epoch = 5.0  # type: ignore[misc]


def test_decaystate_has_no_io():
    """DecayState が os.environ / DB を参照しないこと（純粋性）。"""
    import inspect

    import familiar_agent.time_decay as td

    src = inspect.getsource(td)
    assert "os.environ" not in src
    assert "psycopg" not in src
