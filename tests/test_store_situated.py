"""Tests for store/situated.py（視点ベクトルと situated 行の切り出し）.

`situated_memories` は記憶を「誰の視点から見たか」に寄せたベクトルで、想起の
母集合を作る（[D-在席相関/V2]）。人ごとの視点ベクトル `perspective_vec` と、
埋め込みの平均 `embedding_means`（中心化に使う mu）もここに属する。

このモジュールが持つのは**ベクトルの作成と保存**で、想起（W の構築）は持たない。
想起側の類似検索は S6 で `by_vector` としてストアへ移す。挙動は変えない。
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

from familiar_agent.store.situated import SituatedVectors, _situated_vector
from familiar_agent.tools.memory import ObservationMemory


def test_observation_memory_holds_the_situated_layer() -> None:
    """層は文脈だけで組み立てられ、宿主は部品として持つ（継承しない）。"""
    import inspect

    assert "ctx" in inspect.signature(SituatedVectors.__init__).parameters
    assert not issubclass(ObservationMemory, SituatedVectors)
    assert hasattr(SituatedVectors, "refresh_situated_memories")


def test_situated_vector_is_normalised() -> None:
    """合成したベクトルは長さ1（コサインを取る前提）。"""
    mem = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    per = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    got = _situated_vector(mem, per, None)
    assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-5


def test_situated_vector_subtracts_the_mean_when_given() -> None:
    """mu を渡すと中心化される（渡さなければ従来式）。"""
    mem = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    per = np.zeros(3, dtype=np.float32)
    mu = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert not np.allclose(_situated_vector(mem, per, mu), _situated_vector(mem, per, None))


def test_memory_module_no_longer_writes_situated_rows() -> None:
    """situated への書き込みと perspective_vec / embedding_means が memory.py に無い。

    想起の類似検索（SELECT）は S6 で by_vector として移すため、ここではまだ残る。
    """
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for name in ("perspective_vec", "embedding_means"):
        assert not re.search(rf"\b{name}\b", src), f"{name} が memory.py に残っている"
    assert "INSERT INTO situated_memories" not in src
    assert "UPDATE situated_memories" not in src


def test_situated_module_owns_the_vectors() -> None:
    src = pathlib.Path("src/familiar_agent/store/situated.py").read_text()
    for name in ("situated_memories", "perspective_vec", "embedding_means"):
        assert re.search(rf"\b{name}\b", src), f"{name} が移動先に無い"
