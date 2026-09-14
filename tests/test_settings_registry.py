"""層 3「設定値」の登録の表（記-a-に・2026-09-14・`設計方針_REST内省_設定値を調整する` v0.1 §2）。

登録が無い値は動かせない。登録には範囲・刻み・見る計測（種別と欄）・規則（機械か LLM か）・根拠を
必ず添える。`config_overrides.RANGES` はこの表から導く（二重に持たない）。
"""

from __future__ import annotations

from familiar_agent.config_overrides import RANGES, is_protected
from familiar_agent.core import settings


def test_every_registered_setting_has_range_step_measure_rule_and_note():
    assert len(settings.REGISTRY) >= 26  # distill・HL・far_share・timeout・n×2・内部状態 21
    for s in settings.REGISTRY:
        assert s.lo < s.hi and s.step > 0, s.field
        assert s.measure_kind, s.field
        assert s.rule in ("機械", "LLM"), s.field
        assert s.note.strip(), s.field
        assert not is_protected(s.field), s.field


def test_ranges_are_derived_from_the_registry():
    assert RANGES == {s.field: (s.lo, s.hi) for s in settings.REGISTRY}


def test_the_expected_fields_are_registered():
    fields = {s.field for s in settings.REGISTRY}
    for f in (
        "MemoryConfig.distill_min_a0",
        "MemoryConfig.recall_half_life_days",
        "MemoryConfig.diffuse_far_share",
        "AgentConfig.arbiter_timeout_sec",
        "MemoryConfig.recent_exchanges_arbiter",
        "MemoryConfig.recent_exchanges_main",
        "InnerStateConfig.mood_p_p70",
        "InnerStateConfig.drive_p70_seeking",
    ):
        assert f in fields, f


def test_the_registry_can_be_looked_up_by_field():
    s = settings.get("MemoryConfig.recent_exchanges_main")
    assert s is not None and s.step == 1 and s.rule == "機械" and s.measure_kind == "申告"
    assert settings.get("Nope.x") is None
