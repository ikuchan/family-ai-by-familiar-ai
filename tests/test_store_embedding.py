"""Tests for store/embedding.py（埋め込みの移動）.

埋め込みモデルの遅延読み込みと、ベクトルの符号化・正規化・次元合わせ。想起の
採点でも書き込みでも使うが、SQL も想起ロジックも持たない。`store/` へ移す。
挙動は変えない。
"""

from __future__ import annotations

import numpy as np

from familiar_agent.store.embedding import (
    _coerce_to_embedding_dim,
    _cosine_similarity,
    _decode_vector,
    _encode_vector,
    _EmbeddingModel,
    _normalise,
)


def test_vector_encoding_round_trips() -> None:
    vec = [0.5, -0.25, 0.125]
    back = _decode_vector(_encode_vector(vec))
    assert np.allclose(back, vec, atol=1e-6)


def test_normalise_gives_unit_length() -> None:
    got = _normalise(np.array([3.0, 4.0], dtype=np.float32))
    assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-6


def test_normalise_keeps_zero_vector_finite() -> None:
    got = _normalise(np.zeros(4, dtype=np.float32))
    assert np.all(np.isfinite(got))


def test_cosine_similarity_is_one_for_identical_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    got = _cosine_similarity(v, np.array([v]))
    assert abs(float(got[0]) - 1.0) < 1e-6


def test_coerce_pads_or_truncates_to_the_configured_dim() -> None:
    got = _coerce_to_embedding_dim(np.array([1.0, 2.0], dtype=np.float32))
    from familiar_agent.store.embedding import EMBEDDING_DIM

    assert got.shape[0] == EMBEDDING_DIM


def test_memory_module_uses_the_moved_embedding() -> None:
    """memory.py 側が同じ実体を指している（二重定義になっていない）。"""
    from familiar_agent.tools import memory as memory_module

    assert memory_module._EmbeddingModel is _EmbeddingModel
    assert memory_module._encode_vector is _encode_vector
