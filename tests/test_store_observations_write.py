"""Tests for store/observations.py の書き込み（S6b）.

`observations` と `obs_embeddings` への書き込みを、読み出しと同じファイルへ寄せる。
テーブルの持ち主を1箇所に定めるためで、[D-O書込]（O は追記）の実体がここになる。

この段で `store/jobs.py` が宿主から借りていた `_materialize_save_event` が不要になり、
キュー層は観測層を直接使える。依存が `jobs → observations` の一方向になることを見る。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.observations import ObservationWriteMixin
from familiar_agent.tools.memory import ObservationMemory


def test_observation_memory_inherits_the_write_layer() -> None:
    assert issubclass(ObservationMemory, ObservationWriteMixin)


def test_moved_writers_are_gone_from_memory_module() -> None:
    """移動元に定義が残っていない（二重定義の反証）。"""
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for name in (
        "_materialize_save_event",
        "_mark_recalled",
        "mark_superseded",
        "decay_importance",
    ):
        assert not re.search(rf"^    def {name}\b", src, re.M), f"{name} の定義が残っている"


def test_jobs_no_longer_borrows_materialize_from_the_host() -> None:
    """キュー層の依存が1つ減っている（切り出しが進んだ印）。"""
    src = pathlib.Path("src/familiar_agent/store/jobs.py").read_text()
    assert "宿主が実装する（observations への実体化" not in src


def test_store_dependencies_stay_one_way() -> None:
    """store の中の依存が一方向（循環 import を作らない）。

    clock / db_compat / embedding が葉で、jobs と observations がその上に乗る。
    """
    import familiar_agent.store.jobs  # noqa: F401  import できれば循環していない
    import familiar_agent.store.observations  # noqa: F401

    leaf = pathlib.Path("src/familiar_agent/store/clock.py").read_text()
    assert "from .observations" not in leaf and "from .jobs" not in leaf
