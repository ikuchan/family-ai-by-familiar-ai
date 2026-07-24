"""remember(scope='witnessed') で在席他者ゼロでも記憶を落とさない（floor）。

witnessed 経路は在席他者へのみ書くため、在席他者が居ないと1件も書かれず「書き込みなし」で
content が消えていた（Slice A は speaker 経路だけ floor 済み）。どの scope でも1件も書けなければ
話者／DEFAULT の本命へ floor 書き込みする。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.tools.memory import MemoryTool


def _run(coro):
    return asyncio.run(coro)


def test_witnessed_with_no_others_floors_to_write_store():
    pmm = MagicMock()
    pmm.current_speaker_id = None            # 話者未解決 → DEFAULT
    pmm.get_present_ids.return_value = []
    pmm.get_all_present_memories.return_value = []   # 在席他者なし
    pmm.get_person_name.return_value = "推定話者"
    store = MagicMock()
    store.save_async_with_id = AsyncMock(return_value=("mem-1", True))
    pmm.get_speaker_memory.return_value = None       # _write_store → get_memory_for(DEFAULT)
    pmm.get_memory_for.return_value = store
    tool = MemoryTool(pmm)

    res, _ = _run(tool._remember(
        {"content": "たいきがプールに行った", "scope": "witnessed", "emotion": "happy"}
    ))

    store.save_async_with_id.assert_awaited()   # floor で本命へ書いた
    assert "書き込みなし" not in res


def test_witnessed_with_present_other_writes_to_other_no_floor():
    # 正常系（在席他者あり）：witness の store へ書き、floor は発動しない＝挙動不変
    pmm = MagicMock()
    pmm.current_speaker_id = "SPEAKER"
    pmm.get_present_ids.return_value = ["SPEAKER", "OTHER"]
    other_store = MagicMock()
    other_store.save_async = AsyncMock(return_value=True)
    pmm.get_all_present_memories.return_value = [("OTHER", other_store)]
    pmm.get_person_name.side_effect = lambda pid: {"SPEAKER": "パパ", "OTHER": "ママ"}.get(pid, pid)
    floor_store = MagicMock()
    floor_store.save_async_with_id = AsyncMock(return_value=("m", True))
    pmm.get_speaker_memory.return_value = floor_store
    tool = MemoryTool(pmm)

    res, _ = _run(tool._remember(
        {"content": "何か", "scope": "witnessed", "emotion": "neutral"}
    ))

    other_store.save_async.assert_awaited()          # witness へ書いた
    floor_store.save_async_with_id.assert_not_awaited()  # floor は発動しない（結果あり）
    assert "書き込みなし" not in res
