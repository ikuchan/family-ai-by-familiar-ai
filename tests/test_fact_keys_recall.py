"""記-ad：鍵の文を想起につなぐ（鍵の軸で候補に入れ、鍵の近さを Ruri の頭に入れる）。

測った形（課題8 記-ad「軸と頭」）：鍵の軸は Ruri の軸の後ろに足し、順位の底上げも付ける。記録の Ruri の近さは、
本文の近さと鍵の文のいちばん近い近さの大きいほう。Ruri は conftest の偽物。bge-m3 の問いは固定のベクトル。
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from familiar_agent.core.fact_keys import Fact
from familiar_agent.store.embedding import _RuriModel
from familiar_agent.store.observations import ObservationStore
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_E0 = [1.0] + [0.0] * 1023
_E1 = [0.0, 1.0] + [0.0] * 1022


def _ruri_doc(text: str) -> list[float]:
    from tests.conftest import _HashRuri

    return _HashRuri().encode(["検索文書: " + text])[0].tolist()


@pytest.fixture
def keyed(monkeypatch):
    """A は bge-m3 の問いに近い。C は本文がどの問いにも遠く、鍵の文だけが問いに一致する。Ruri の本文の軸は外す。"""
    monkeypatch.setenv("RECALL_PRIMARY_N", "1")
    monkeypatch.setenv("RECALL_W_T", "0")
    monkeypatch.setenv("RECALL_W_E", "0")
    monkeypatch.setattr(ObservationStore, "by_ruri", lambda self, *a, **k: [])
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    tag = uuid.uuid4().hex
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_E0]):
        ida, _ = mem.save_with_id("近い本文 " + tag, materialize_now=True)
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_E1]):
        idc, _ = mem.save_with_id("遠い本文 " + tag, materialize_now=True)
    key = f"町内会{tag}の夏祭りの日"
    return mem, ida, idc, tag, key


def _write_key(mem, obs_id, tag):
    srcs = [s for s in mem.fact_pending(1_000_000) if obs_id in s.obs_ids]
    assert mem.write_fact_keys(srcs, [[Fact("町内会" + tag, "夏祭りの日", "8月3日")]], model="m")


def _recall(mem, key):
    with (
        patch.object(_EmbeddingModel, "encode_query", return_value=[_E0]),
        patch.object(_RuriModel, "encode_query", return_value=[_ruri_doc(key)]),
    ):
        return mem.recall("夏祭りっていつ？", n=5, min_score=0.0)


def test_the_key_axis_brings_a_record_only_its_key_sentence_finds(keyed) -> None:
    mem, ida, idc, tag, key = keyed
    _write_key(mem, idc, tag)
    got = [r["memory_id"] for r in _recall(mem, key)]
    assert idc in got and ida in got


def test_without_key_sentences_the_record_is_not_found(keyed) -> None:
    mem, ida, idc, _, key = keyed
    got = [r["memory_id"] for r in _recall(mem, key)]
    assert got == [ida]


def test_the_key_nearness_goes_into_the_ruri_head(keyed, monkeypatch) -> None:
    """ほかの軸で拾った記録も、鍵の近さが本文の Ruri の近さより大きければ、それが採点に渡る Ruri の近さになる。

    A（bge-m3 の軸で拾う）に鍵の文を持たせ、鍵の軸は外して、頭に入る近さ（`fact_key_cosines`）だけを比べる。
    A の本文の Ruri の近さは、問い（鍵の文と同じ点）とほぼ 0 で、鍵の文の近さは 1。
    """
    from familiar_agent.tools import memory as m

    mem, ida, idc, tag, key = keyed
    _write_key(mem, ida, tag)
    monkeypatch.setattr(ObservationStore, "by_fact_keys", lambda self, *a, **k: [])
    seen: list = []
    real = m._score_breakdown

    def spy(*a, **k):
        seen.append(k.get("ruri_cosine"))
        return real(*a, **k)

    monkeypatch.setattr(m, "_score_breakdown", spy)
    _recall(mem, key)
    assert max(c for c in seen if c is not None) > 0.99
    seen.clear()
    monkeypatch.setattr(ObservationStore, "fact_key_cosines", lambda self, *a, **k: {})
    _recall(mem, key)
    assert all(c is None or c < 0.99 for c in seen)


def test_a_broken_key_axis_degrades_to_the_old_recall(keyed) -> None:
    """鍵の表が引けなくても（次元違いなど）、想起はいままでどおり続く。"""
    mem, ida, idc, tag, _ = keyed
    _write_key(mem, idc, tag)
    with (
        patch.object(_EmbeddingModel, "encode_query", return_value=[_E0]),
        patch.object(_RuriModel, "encode_query", return_value=[[1.0, 2.0]]),
    ):
        got = [r["memory_id"] for r in mem.recall("夏祭りっていつ？", n=5, min_score=0.0)]
    assert got == [ida]
