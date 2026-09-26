"""試験の非同期の待ちが止まったら、何を待っていたかを残して失敗する（環-aa・2026-09-26）。

全体テストが 14 回中 3 回、`test_event_loop.py` の別々の試験で 120 秒の上限に達して止まった。止まったときに
生きていたのはメインのスレッド（何もせず待っている）だけで、**どのタスクが何を待っていたかは残らなかった**。
再現を狙って 9 回回しても出なかった。次に止まったとき必ず原因が残るよう、`conftest.py` の見張り
（`_async_stuck_guard`）が `asyncio.run` を包む。秒数は `TEST_ASYNC_STUCK_SEC`（既定 90・上限 120 より短く）。
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.timeout(5)
def test_a_stuck_run_fails_with_the_waiting_tasks(monkeypatch):
    monkeypatch.setenv("TEST_ASYNC_STUCK_SEC", "1")

    async def waits_forever():
        await asyncio.Event().wait()  # 誰も set しない

    with pytest.raises(TimeoutError) as err:
        asyncio.run(waits_forever())
    assert "waits_forever" in str(err.value)  # 待っていたタスクの場所が載る


def test_a_run_that_finishes_is_untouched():
    async def answers():
        await asyncio.sleep(0)
        return 42

    assert asyncio.run(answers()) == 42
