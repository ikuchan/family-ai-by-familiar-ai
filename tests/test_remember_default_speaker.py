"""Slice A：話者未解決でも書き込みは既定話者 DEFAULT へフォールバックする。

単独テキスト（声/カメラ/明示のどれも無い）で remember が「話者不明／書き込みなし」に
ならず、DEFAULT_PERSON_ID の記憶へ帰属する。優先度の floor（声>カメラ>明示>default）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.tools.memory import MemoryTool


def test_write_store_falls_back_to_default_person_when_no_speaker():
    pmm = MagicMock()
    pmm.get_speaker_memory.return_value = None  # 話者未解決
    sentinel = object()
    pmm.get_memory_for.return_value = sentinel
    tool = MemoryTool(pmm)

    assert tool._write_store is sentinel  # None でなく DEFAULT person の記憶へ
    pmm.get_memory_for.assert_called_once_with(DEFAULT_PERSON_ID)


def test_write_store_uses_speaker_when_present():
    pmm = MagicMock()
    speaker_mem = object()
    pmm.get_speaker_memory.return_value = speaker_mem  # 話者あり（声/カメラ/明示）
    tool = MemoryTool(pmm)

    assert tool._write_store is speaker_mem  # 話者がいればそちら優先
    pmm.get_memory_for.assert_not_called()
