"""話者の名前の器（`PersonRegistry`）は名前だけを持つ（関係の値の器は環-d で外した・2026-09-15）。"""

from __future__ import annotations


def test_the_person_registry_keeps_only_names():
    from familiar_agent.relationship import PersonRegistry

    reg = PersonRegistry(default_name="パジュ")
    assert reg.active_name == "パジュ" and not reg.active_is_explicit
    reg.register("パパ")
    reg.set_active("パパ")
    assert reg.active_name == "パパ" and reg.active_is_explicit
    reg.reset_to_default()
    assert reg.active_name == "パジュ"
