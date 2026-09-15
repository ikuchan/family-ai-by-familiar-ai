"""旧 15 欲求 `DesireSystem` の撤去（環-d・2026-09-15）。

`agent.run()` は `desires`・`desire_name` を受けず、GUI／TUI も渡さない。欲求は 5 軸だけ。
旧名は `src/`・`tests/` から消える（数え上げでなく grep で 0 件）。
"""

from __future__ import annotations

import inspect
import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"
_OLD = (
    "DesireSystem",
    "is_internal_desire_turn",
    "detect_worry_signal",
    "is_social_desire",
    "AdaptiveDesireCooldown",
    "should_fire_idle_desire",
    "desire_tick_prompt",
    "extract_curiosity",
    "_desires_ref",
    "is_desire_turn",
)


def test_agent_run_takes_no_desire_arguments():
    from familiar_agent.agent import EmbodiedAgent

    params = inspect.signature(EmbodiedAgent.run).parameters
    assert "desires" not in params and "desire_name" not in params


def test_no_old_name_survives_in_the_sources():
    hits = []
    for f in _SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for name in _OLD:
            for m in re.finditer(rf"\b{name}\b", text):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{f.relative_to(_SRC)}:{line} {name}")
    assert not hits, "旧欲求の名前が残っている:\n" + "\n".join(hits)
    assert not (_SRC / "desires.py").exists()
