"""「整合チェック」を「発話前の検査」へ改名する（出-ag-ろ・2026-09-25）。

中身は**言おうとしている返事が規則に反していないかを、発話の前に見る**ことで、何かと何かの
「整合」を見ているのではない。名前が中身と合わず、本人から「何と何の整合性か」と問われた。
名前は本人が決めた（「発話前の検査」・英語は `speech_check`）。
"""

from __future__ import annotations

import importlib

import pytest


def test_the_module_has_the_new_name():
    mod = importlib.import_module("familiar_agent.loop.speech_check")
    assert callable(mod.facts_ctx)
    assert callable(mod.used_lines)


def test_the_old_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("familiar_agent.loop.coherence")


def test_the_evaluator_checks_speech():
    from familiar_agent.loop.evaluator import Evaluator

    assert callable(getattr(Evaluator, "check_speech", None))
    assert not hasattr(Evaluator, "check_response_coherence")


def test_the_loop_checks_speech():
    from familiar_agent.loop.event_loop import InformationProcessing

    assert callable(getattr(InformationProcessing, "_speech_check_violation", None))
    assert not hasattr(InformationProcessing, "_coherence_violation")


def test_the_switch_has_the_new_name(monkeypatch):
    from familiar_agent.config import AgentConfig

    monkeypatch.setenv("FAMILIAR_SPEECH_CHECK", "0")
    cfg = AgentConfig()
    assert cfg.speech_check is False
    assert not hasattr(cfg, "coherence_check")


def test_no_old_name_is_left_in_the_source():
    """旧名が残っていない（大文字小文字を問わない）。

    段 1 の検索は大文字小文字を区別していて、ログの文「Coherence check …」と、表記の違う
    「整合性チェック」を取りこぼした。名前の網羅は、数え上げでなく検索 0 件で確かめる。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    old = re.compile(r"coherence|整合性?チェック", re.I)
    left = [
        f"{p.relative_to(src)}:{i}: {line.strip()[:60]}"
        for p in sorted(src.rglob("*.py"))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if old.search(line)
    ]
    assert left == [], "\n".join(left)
