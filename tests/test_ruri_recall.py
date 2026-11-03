"""記-y B 段：Ruri を想起につなぐ（軸・r の大きいほう・どちらの頭かのログ）。

Ruri は conftest の偽物（文から決まる 768 個の数値）。bge-m3 の問いは固定のベクトルに差し替える。
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import patch

import pytest

from familiar_agent.store.embedding import _RuriModel
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_E0 = [1.0] + [0.0] * 1023
_E1 = [0.0, 1.0] + [0.0] * 1022


def _ruri_doc(text: str) -> list[float]:
    from tests.conftest import _HashRuri

    return _HashRuri().encode(["検索文書: " + text])[0].tolist()


@pytest.fixture
def two_records(monkeypatch):
    """A は bge-m3 の問いに近く、B は Ruri の問いにだけ近い。軸は bge-m3 と Ruri だけにする。"""
    monkeypatch.setenv("RECALL_PRIMARY_N", "1")
    monkeypatch.setenv("RECALL_W_T", "0")
    monkeypatch.setenv("RECALL_W_E", "0")
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    a, b = ("ruri recall a " + uuid.uuid4().hex), ("ruri recall b " + uuid.uuid4().hex)
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_E0]):
        ida, _ = mem.save_with_id(a, materialize_now=True)
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_E1]):
        idb, _ = mem.save_with_id(b, materialize_now=True)
    return mem, ida, idb, b


def _recall(mem, ruri_q, **kw):
    with (
        patch.object(_EmbeddingModel, "encode_query", return_value=[_E0]),
        patch.object(_RuriModel, "encode_query", return_value=ruri_q),
    ):
        return mem.recall("qqq zzz", n=5, **kw)


def test_ruri_axis_brings_a_record_only_ruri_finds(two_records) -> None:
    """bge-m3 の軸（1 件）では拾えない B が、Ruri の軸で候補に入る。"""
    mem, ida, idb, b = two_records
    got = [r["memory_id"] for r in _recall(mem, [_ruri_doc(b)])]
    # 順位は問わない：A は bge-m3 と Ruri の両方の軸に載るので、順位の加点が B より多い（設計どおり）。
    assert ida in got and idb in got


def test_without_ruri_the_record_is_not_found(two_records) -> None:
    """Ruri が休み（問いを数値にできない）なら、いままでどおり bge-m3 だけで想起する。"""
    mem, ida, idb, _ = two_records
    got = [r["memory_id"] for r in _recall(mem, None)]
    assert got == [ida]


def test_info_line_counts_heads(two_records, caplog) -> None:
    mem, ida, idb, b = two_records
    with caplog.at_level(logging.INFO, logger="familiar_agent.tools.memory"):
        _recall(mem, [_ruri_doc(b)])
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    heads = [ln for ln in lines if ln.startswith("想起 上位")]
    assert heads == ["想起 上位 2 件の頭：bge-m3 1・Ruri 1"], lines
    # 記憶の中身は info に出さない。
    assert all(b not in ln for ln in heads)


def test_info_line_says_ruri_is_resting(two_records, caplog) -> None:
    mem, *_ = two_records
    with caplog.at_level(logging.INFO, logger="familiar_agent.tools.memory"):
        _recall(mem, None)
    heads = [r.getMessage() for r in caplog.records if r.getMessage().startswith("想起 上位")]
    assert heads == ["想起 上位 1 件の頭：bge-m3 1（Ruri 休み）"]


def test_debug_line_carries_head_and_both_values(two_records, caplog) -> None:
    mem, ida, idb, b = two_records
    with caplog.at_level(logging.DEBUG, logger="familiar_agent.tools.memory"):
        _recall(mem, [_ruri_doc(b)])
    rows = [r.getMessage() for r in caplog.records if "recall score=" in r.getMessage()]
    assert any("頭=Ruri" in ln and "r_ruri=1.000" in ln for ln in rows), rows
    assert any("頭=bge-m3" in ln for ln in rows), rows


def test_info_line_says_resting_when_ruri_cannot_be_read(two_records, caplog) -> None:
    """問いは数値にできても、DB から Ruri を引けなければ（表が壊れた・次元違いなど）「休み」と出す。
    「Ruri 0」と出すと、Ruri が動いて負けたように見える。"""
    from familiar_agent.store.observations import ObservationStore

    mem, ida, idb, b = two_records
    with (
        caplog.at_level(logging.INFO, logger="familiar_agent.tools.memory"),
        patch.object(ObservationStore, "by_ruri", return_value=[]),
        patch.object(ObservationStore, "ruri_cosines", return_value={}),
    ):
        got = [r["memory_id"] for r in _recall(mem, [_ruri_doc(b)])]
    assert got == [ida]
    heads = [r.getMessage() for r in caplog.records if r.getMessage().startswith("想起 上位")]
    assert heads == ["想起 上位 1 件の頭：bge-m3 1（Ruri 休み）"]


def test_ruri_does_not_replace_the_row_another_axis_found(two_records, monkeypatch) -> None:
    """ほかの軸がすでに拾った記録の行（代表の面）は、Ruri の軸が置き換えない。Ruri の軸は最後に足す。"""
    from familiar_agent.store.observations import ObservationStore

    mem, ida, idb, b = two_records
    real = ObservationStore.by_ruri

    def marked(self, *a, **kw):
        return [dict(r, content="FROM-RURI") for r in real(self, *a, **kw)]

    monkeypatch.setenv("RECALL_W_T", "1")
    with patch.object(ObservationStore, "by_ruri", marked):
        res = _recall(mem, [_ruri_doc(b)])
    by_id = {r["memory_id"]: r for r in res}
    # A と B は時間の軸でも拾われるので、行は時間の軸のもの。
    assert by_id[ida]["summary"] != "FROM-RURI"
    assert by_id[idb]["summary"] != "FROM-RURI"
