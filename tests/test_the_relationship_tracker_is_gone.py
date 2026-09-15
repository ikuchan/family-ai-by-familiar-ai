"""`RelationshipTracker`（`relationship_state`）の撤去（環-d・2026-09-15）。

関係の値（trust・intimacy）は毎ターン書かれていたが、読むのは呼び手の無い宛先選びだけだった。
`PersonRegistry`（話者の名前の器）は残る。旧名は grep で 0 件。
"""

from __future__ import annotations

import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"
_OLD = (
    "RelationshipTracker",
    "relationship_state",
    "record_conversation",
    "_select_addressee",
    "_proactive_memory_context",
    "weight_relation",
    "_relationship",
)


def test_no_old_name_survives_in_the_sources():
    hits = []
    for f in _SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for name in _OLD:
            for m in re.finditer(rf"\b{name}\b", text):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{f.relative_to(_SRC)}:{line} {name}")
    assert not hits, "旧名が残っている:\n" + "\n".join(hits)


def test_the_person_registry_keeps_only_names():
    from familiar_agent.relationship import PersonRegistry

    reg = PersonRegistry(default_name="パジュ")
    assert reg.active_name == "パジュ" and not reg.active_is_explicit
    reg.register("パパ")
    reg.set_active("パパ")
    assert reg.active_name == "パパ" and reg.active_is_explicit
    reg.reset_to_default()
    assert reg.active_name == "パジュ"
