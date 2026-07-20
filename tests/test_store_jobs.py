"""Tests for store/jobs.py（非同期の書き込みキューの切り出し）.

記憶の書き込みは、いきなり `observations` へ入らず、いったんイベントとして積んで
から実体化される（[D-O書込]：O は追記＝イベントログ）。重い処理を応答の経路から
外すための仕組みでもある。

    save(materialize_now=False)
      → memory_events に追記    （何を書くか）
      → memory_jobs に積む      （いつ実体化するか）
      → claim_pending_jobs で拾って materialize → observations に現れる

この2テーブルの持ち主を `store/jobs.py` に定める。Phase 3（知覚）が書き込み経路へ
話者帰属を供給する前に、キューの所有権を1箇所へ寄せておく。挙動は変えない。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.jobs import JobQueue
from familiar_agent.tools.memory import ObservationMemory


def test_observation_memory_holds_the_jobs_layer() -> None:
    """公開 API は変わらない（memory_worker.py を書き換えないための委譲）。"""
    import inspect

    assert "observations" in inspect.signature(JobQueue.__init__).parameters
    assert hasattr(ObservationMemory, "claim_pending_jobs")


def test_worker_entry_points_are_still_reachable() -> None:
    """memory_worker.py が呼ぶ入口が残っている。"""
    for name in (
        "claim_pending_jobs",
        "mark_job_done",
        "mark_job_failed",
        "materialize_event",
        "append_memory_event",
    ):
        assert hasattr(ObservationMemory, name), name


def test_memory_module_no_longer_owns_the_queue_tables() -> None:
    """memory_events / memory_jobs の SQL が memory.py に残っていない。"""
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for table in ("memory_events", "memory_jobs"):
        assert not re.search(rf"\b{table}\b", src), f"{table} が memory.py に残っている"


def test_jobs_module_owns_the_queue_tables() -> None:
    """移動先がそのテーブルを持っている（空でない）。"""
    src = pathlib.Path("src/familiar_agent/store/jobs.py").read_text()
    for table in ("memory_events", "memory_jobs"):
        assert re.search(rf"\b{table}\b", src), f"{table} が移動先に無い"
