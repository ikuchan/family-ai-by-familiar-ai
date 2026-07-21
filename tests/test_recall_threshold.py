"""Tests for recall() relevance-score threshold filter.

recall(min_score=X) は合成 final score が X 以上の記憶だけを返す（生コサインではない）。
RECALL_MIN_SCORE env var is read via MemoryConfig.recall_min_score（既定 0.05）。
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory():
    """Minimal ObservationMemory with mocked embedder (no ML model needed)."""
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


@pytest.fixture()
def memory_with_data(memory):
    """ObservationMemory pre-loaded with one observation."""
    memory.save("関連する内容についての観察", kind="observation", emotion="curious")
    return memory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recall_accepts_min_score_param(memory):
    """recall() accepts min_score keyword argument without error."""
    result = memory.recall("テスト", n=3, min_score=0.5)
    assert isinstance(result, list)


def test_recall_default_min_score_is_zero(memory_with_data):
    """min_score omitted and min_score=0.0 return the same results (backward-compat)."""
    result_default = memory_with_data.recall("テスト", n=5)
    result_zero = memory_with_data.recall("テスト", n=5, min_score=0.0)
    assert len(result_default) == len(result_zero)


def test_recall_filters_below_threshold(memory_with_data):
    """min_score filters out memories below the threshold."""
    # With encode_query returning [1,0,0] and encode_document returning [1,0,0],
    # cosine similarity = 1.0, so high threshold should still return the memory
    high = memory_with_data.recall("関連する内容", n=10, min_score=0.5)
    assert all(m["score"] >= 0.5 for m in high)


def test_recall_high_threshold_excludes_all(memory_with_data):
    """min_score=1.01 (impossible) excludes everything."""
    results = memory_with_data.recall("関連する内容", n=10, min_score=1.01)
    assert results == []


def test_recall_min_score_from_env(monkeypatch, memory):
    """RECALL_MIN_SCORE env var is read via MemoryConfig."""
    monkeypatch.setenv("RECALL_MIN_SCORE", "0.6")
    from familiar_agent.config import MemoryConfig
    assert MemoryConfig().recall_min_score == pytest.approx(0.6)


def test_recall_min_score_env_default(monkeypatch):
    """MemoryConfig.recall_min_score returns 0.05 (起点) when RECALL_MIN_SCORE is unset."""
    monkeypatch.delenv("RECALL_MIN_SCORE", raising=False)
    from familiar_agent.config import MemoryConfig
    assert MemoryConfig().recall_min_score == pytest.approx(0.05)


def test_recall_min_score_env_invalid(monkeypatch):
    """MemoryConfig.recall_min_score falls back to 0.05 on invalid env value."""
    monkeypatch.setenv("RECALL_MIN_SCORE", "not-a-number")
    from familiar_agent.config import MemoryConfig
    assert MemoryConfig().recall_min_score == pytest.approx(0.05)
