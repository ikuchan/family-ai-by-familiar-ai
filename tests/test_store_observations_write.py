"""Tests for store/observations.py の書き込み（S6b）.

`observations` と `obs_embeddings` への書き込みを、読み出しと同じファイルへ寄せる。
テーブルの持ち主を1箇所に定めるためで、[D-O書込]（O は追記）の実体がここになる。

この段で `store/jobs.py` が宿主から借りていた `_materialize_save_event` が不要になり、
キュー層は観測層を直接使える。依存が `jobs → observations` の一方向になることを見る。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.observations import ObservationStore


def test_observation_memory_holds_the_write_layer() -> None:
    assert hasattr(ObservationStore, "materialize_save_event")


def test_writing_is_owned_by_the_store_layer() -> None:
    """書き込みの実装が層にあり、`memory.py` に二重に無い。"""
    from familiar_agent.store.observations import ObservationStore

    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for name in ("materialize_save_event", "_mark_recalled"):
        assert hasattr(ObservationStore, name), f"{name} が層に無い"
    for name in ("_materialize_save_event", "_mark_recalled"):
        assert not re.search(rf"^    (?:async )?def {name}\b", src, re.M), \
            f"{name} が memory.py にも定義されている（二重実装）"

    # 外から呼ばれるものは委譲が残る。ただし SQL は持たない。
    for name in ("mark_superseded",):
        m = re.search(rf"^    (?:async )?def {name}\b.*?(?=^    (?:async )?def |\Z)", src, re.M | re.S)
        assert m, f"{name} の委譲が要る（外から呼ばれる）"
        assert "cur.execute" not in m.group(0), f"{name} の SQL が memory.py に残っている"


def test_jobs_receives_observations_explicitly() -> None:
    """キュー層は実体化の本体を引数で受け取る（借り物でない）。"""
    import inspect

    from familiar_agent.store.jobs import JobQueue

    assert "observations" in inspect.signature(JobQueue.__init__).parameters


def test_store_dependencies_stay_one_way() -> None:
    """store の中の依存が一方向（循環 import を作らない）。

    clock / db_compat / embedding が葉で、jobs と observations がその上に乗る。
    """
    import familiar_agent.store.jobs  # noqa: F401  import できれば循環していない
    import familiar_agent.store.observations  # noqa: F401

    leaf = pathlib.Path("src/familiar_agent/store/clock.py").read_text()
    assert "from .observations" not in leaf and "from .jobs" not in leaf
