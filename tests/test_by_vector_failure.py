"""by_vector の失敗を握り潰さず recall まで上げる（棚卸し A1）。

dumb 層 `by_vector` は失敗を例外として上げ（従来は握り潰して `[]`）、方針は recall が
持つ。recall は失敗時に loud（`logger.exception`）に残して `[]` を返し、**失敗時は
keyword_fallback へ流さない**（「壊れているのにテキスト検索で動いて見える」masking を消す）。
正当な0件（situated 未生成）は by_vector が正常に空を返す経路なので keyword_fallback は不変。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ── by_vector：失敗を握り潰さず送出する（従来は []） ──────────────────────────

def test_by_vector_raises_on_query_error():
    from familiar_agent.store.observations import ObservationStore

    store = ObservationStore.__new__(ObservationStore)
    ctx = MagicMock()
    ctx.person_id = "p"
    ctx.conn.return_value.cursor.side_effect = RuntimeError("boom")
    store._ctx = ctx

    with pytest.raises(RuntimeError):
        store.by_vector("q", 5)


# ── recall：by_vector 失敗時に [] を返し keyword_fallback を呼ばない ─────────

def _fixed_embed():
    return (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    )


def test_recall_returns_empty_and_skips_keyword_on_by_vector_failure(caplog):
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        with patch.object(mem._observations, "by_vector", side_effect=RuntimeError("boom")), \
             patch.object(mem._observations, "keyword_fallback") as kf, \
             caplog.at_level(logging.ERROR, logger="familiar_agent.tools.memory"):
            res = mem.recall("anything", n=3)  # min_score 既定 0.0（従来はここで keyword へ流れた）
        assert res == []
        kf.assert_not_called()  # 失敗を keyword 検索で masking しない
        # 失敗はトレース付きで loud に残る（warning でなく error/exception）。
        assert any(
            r.levelno >= logging.ERROR and "recall failed" in r.getMessage()
            for r in caplog.records
        )
    finally:
        for p in ps:
            p.stop()
