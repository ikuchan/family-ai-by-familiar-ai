"""Tests for legacy/semantic_layer.py（撤去予定の意味・信念層の隔離）.

`semantic_facts`／`behavior_policies`／`memory_revisions`／`memory_links` は
Phase 6 で撤去が決まっている（課題8）。設計上は、意味・信念は O へ一元化され、
明示リンクは WR 拡散想起へ置き換わる（[D-記憶単一化]／[D-WR拡散想起]）。

撤去予定のものを1ファイルへ隔離しておけば、Phase 6 の撤去は「ファイルを消して
基底クラスから1語外す」に近づく（Parnas：変わりそうな判断を隠す単位で切る）。

ここでは移動だけを行い、挙動は変えない。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.legacy.semantic_layer import LegacySemanticLayer
from familiar_agent.tools.memory import ObservationMemory


_LEGACY_TABLES = ("semantic_facts", "behavior_policies", "memory_revisions", "memory_links")


def test_observation_memory_holds_the_legacy_layer() -> None:
    """公開 API は変わらない（呼び出し側を書き換えないための委譲）。"""
    import inspect

    assert "ctx" in inspect.signature(LegacySemanticLayer.__init__).parameters
    assert hasattr(ObservationMemory, "recall_semantic_facts_async")


def test_public_methods_are_still_reachable() -> None:
    """agent.py が呼んでいる入口が残っている。"""
    for name in (
        "recall_semantic_facts_async",
        "recall_behavior_policies_async",
        "adjust_semantic_fact_confidence_async",
        "adjust_behavior_policy_confidence_async",
        "link_memories_async",
        "get_linked_memories_async",
        "format_semantic_facts_for_context",
        "format_behavior_policies_for_context",
    ):
        assert hasattr(ObservationMemory, name), name


def test_memory_module_no_longer_mentions_the_legacy_tables() -> None:
    """撤去予定のテーブル名が memory.py に残っていない（隔離の完了条件）。"""
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for table in _LEGACY_TABLES:
        assert not re.search(rf"\b{table}\b", src), f"{table} が memory.py に残っている"


def test_legacy_module_owns_the_legacy_tables() -> None:
    """移した先に、それらのテーブルの SQL がある（移動先が空でない）。"""
    src = pathlib.Path("src/familiar_agent/legacy/semantic_layer.py").read_text()
    for table in _LEGACY_TABLES:
        assert re.search(rf"\b{table}\b", src), f"{table} が移動先に無い"
