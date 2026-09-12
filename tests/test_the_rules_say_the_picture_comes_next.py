"""主LLM への規則と目の説明が、実態（写真は次の反復に届く）と合っている（`イベント駆動ループ` v0.43）。

「この反復で見た画像に写っていたことだけ」は、画像が次の反復に来る設計では満たしようがなく、
主LLM は see を出し直し続けた（2026-09-12 実機・止められた see 3回）。
"""

from __future__ import annotations

from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT, drop_constraint, rules_section


def _rule(rule_id: str) -> str:
    text = rules_section()
    at = text.index(f":id {rule_id}")
    return text[at : text.index('")', at)]


def test_no_fake_perception_speaks_of_the_received_picture() -> None:
    rule = _rule("no-fake-perception")
    assert "受け取った写真" in rule
    assert "出し直さず" in rule
    assert "この反復で" not in rule


def test_the_eyes_say_the_picture_arrives_next_iteration() -> None:
    at = EVENT_SYSTEM_PROMPT.index(":id eyes")
    desc = EVENT_SYSTEM_PROMPT[at : EVENT_SYSTEM_PROMPT.index("\n", at)]
    assert "次の反復" in desc and "写真" in desc


def test_the_rule_can_still_be_dropped() -> None:
    assert "no-fake-perception" not in drop_constraint(EVENT_SYSTEM_PROMPT, "no-fake-perception")


def test_the_see_tool_says_not_to_call_again() -> None:
    from familiar_agent.tools.camera import CameraTool

    defs = CameraTool.get_tool_definitions(_Stub())
    see = next(d for d in defs if d["name"] == "see")
    assert "do not call see() again" in see["description"]


class _Stub:
    _poses: list = []
