"""save 系の失敗を loud に残す（棚卸し A4）。

`save`/`save_with_id` は失敗を握り潰して `False`/`None` を返すが、その痕跡が
トレース無しの warning だと、決定的エラー（埋め込み次元不一致・モデル未ロード・
コードバグ等）が沈黙する。返り値は従来どおり（ターンは落とさない）まま、失敗を
`logger.exception`（トレース付き・error 相当）で残す。
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


def _fixed_embed():
    return (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    )


def test_save_failure_is_loud_error_with_trace(caplog):
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        with patch.object(
            mem._observations, "materialize_save_event", side_effect=RuntimeError("boom")
        ), caplog.at_level(logging.ERROR, logger="familiar_agent.tools.memory"):
            ok = mem.save("save loud x", kind="observation")
        assert ok is False  # 返りは従来どおり（ターンを落とさない）
        assert any(
            r.levelno >= logging.ERROR and "save failed" in r.getMessage() and r.exc_info
            for r in caplog.records
        ), "save 失敗が error＋トレースで残っていない"
    finally:
        for p in ps:
            p.stop()


def test_save_with_id_failure_is_loud_error_with_trace(caplog):
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        with patch.object(
            mem._observations, "materialize_save_event", side_effect=RuntimeError("boom")
        ), caplog.at_level(logging.ERROR, logger="familiar_agent.tools.memory"):
            mem_id, ok = mem.save_with_id("save loud y", kind="observation")
        assert (mem_id, ok) == (None, False)
        assert any(
            r.levelno >= logging.ERROR and "save_with_id failed" in r.getMessage() and r.exc_info
            for r in caplog.records
        )
    finally:
        for p in ps:
            p.stop()
