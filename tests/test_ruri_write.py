"""記-y A5：記録を書くとき・本文に書き足すとき、Ruri の数値も `obs_ruri_embeddings` に書く。

Ruri の数値は記録の本文から作る（記録 1 件に 1 つ）。数値にできなければ行を書かない（0 は書かない）。
Ruri は conftest の偽物（文から決まる数値）で、本物は読み込まない。
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.store.embedding import _RuriModel
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_DB_URL = os.environ["DATABASE_URL"]
_FIXED = [1.0] + [0.0] * 1023


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _ruri_row(obs_id: str) -> "np.ndarray | None":
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT vector::text AS v FROM obs_ruri_embeddings WHERE obs_id = %s", (obs_id,)
            )
            r = cur.fetchone()
    finally:
        conn.close()
    if r is None:
        return None
    return np.array(r["v"].strip("[]").split(","), dtype=np.float32)


def _expected(text: str) -> np.ndarray:
    from tests.conftest import _HashRuri

    return _HashRuri().encode(["検索文書: " + text])[0]


def test_saving_writes_the_ruri_vector() -> None:
    body = "ruri write " + uuid.uuid4().hex
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        obs_id, _ = _mem().save_with_id(body, materialize_now=True)
    assert obs_id
    v = _ruri_row(obs_id)
    assert v is not None and v.shape == (768,)
    assert np.allclose(v, _expected(body), atol=1e-5)


def test_ruri_failure_writes_no_row_but_keeps_the_record() -> None:
    body = "ruri fail " + uuid.uuid4().hex
    with (
        patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]),
        patch.object(_RuriModel, "encode_document", return_value=None),
    ):
        obs_id, _ = _mem().save_with_id(body, materialize_now=True)
    assert obs_id
    assert _ruri_row(obs_id) is None


def test_appending_rewrites_the_ruri_vector() -> None:
    body = "ruri append " + uuid.uuid4().hex
    mem = _mem()
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        obs_id, _ = mem.save_with_id(body, materialize_now=True)
        assert mem._observations.append_and_reembed(obs_id, "あとから足した一行") is True
    v = _ruri_row(obs_id)
    assert v is not None
    assert np.allclose(v, _expected(f"{body}\nあとから足した一行"), atol=1e-5)


# ── A6：Ruri で拾う読み出しと、記録ごとの Ruri の近さ ──────────────────────


def _q(text: str) -> str:
    """偽物の Ruri で「その本文と同じ数値」の問いを作る（近さ 1 になる）。"""
    from familiar_agent.db import vec_to_sql

    return vec_to_sql(_expected(text).tolist())


_BGE_Q = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def test_by_ruri_ranks_by_ruri_and_skips_hidden_and_excluded() -> None:
    mem = _mem()
    a, b, c = (f"ruri axis {k} " + uuid.uuid4().hex for k in "abc")
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        ida, _ = mem.save_with_id(a, materialize_now=True)
        idb, _ = mem.save_with_id(b, materialize_now=True)
        idc, _ = mem.save_with_id(c, materialize_now=True)
    obs = mem._observations
    rows = obs.by_ruri(_q(a), _BGE_Q, 10)
    assert rows[0]["id"] == ida
    assert abs(rows[0]["score"] - 1.0) < 1e-4
    # 1 記録 1 行（代表の面に畳む）。
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    # 想起の行と同じ列を持つ（by_vector の行と混ぜて採点するため）。
    for k in ("facet_id", "timestamp", "groundedness_g0", "groundedness_n", "content", "direction"):
        assert k in rows[0]
    # 隠された記録（旧）は拾わない。
    assert obs.mark_superseded(idb, idc) is True
    assert idb not in [r["id"] for r in obs.by_ruri(_q(b), _BGE_Q, 10)]
    # 除外の指定を守る。
    assert ida not in [r["id"] for r in obs.by_ruri(_q(a), _BGE_Q, 10, exclude_ids=[ida])]


def test_ruri_cosines_for_given_records() -> None:
    mem = _mem()
    a = "ruri cos a " + uuid.uuid4().hex
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        ida, _ = mem.save_with_id(a, materialize_now=True)
        with patch.object(_RuriModel, "encode_document", return_value=None):
            idn, _ = mem.save_with_id("no ruri " + uuid.uuid4().hex, materialize_now=True)
    cos = mem._observations.ruri_cosines(_q(a), [ida, idn])
    assert abs(cos[ida] - 1.0) < 1e-4
    # Ruri の数値が無い記録は結果に含めない（呼び出し側で bge-m3 だけにする）。
    assert idn not in cos


# ── 見直し 1 回目の指摘 ─────────────────────────────────────────────────────


def test_a_broken_ruri_write_does_not_lose_the_record() -> None:
    """Ruri の行が書けなくても（次元違いなど）、記録・面は残る。Ruri は縮退の側。"""
    from familiar_agent.store.observations import ObservationStore

    body = "ruri broken write " + uuid.uuid4().hex
    with (
        patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]),
        patch.object(ObservationStore, "_ruri_sql", return_value="[1,2]"),
    ):
        obs_id, _ = _mem().save_with_id(body, materialize_now=True)
    assert obs_id
    conn = psycopg2.connect(_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM observations WHERE id = %s", (obs_id,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM situated_memories WHERE obs_id = %s", (obs_id,))
            assert cur.fetchone()[0] >= 1
    finally:
        conn.close()
    assert _ruri_row(obs_id) is None


def test_append_with_ruri_failure_drops_the_stale_row() -> None:
    """書き足しで Ruri が数値にできなければ、古い本文の行を消す（行が無い＝数値が無い、を保つ）。"""
    body = "ruri stale " + uuid.uuid4().hex
    mem = _mem()
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        obs_id, _ = mem.save_with_id(body, materialize_now=True)
        assert _ruri_row(obs_id) is not None
        with patch.object(_RuriModel, "encode_document", return_value=None):
            assert mem._observations.append_and_reembed(obs_id, "足した一行") is True
    assert _ruri_row(obs_id) is None


def test_by_ruri_failure_degrades_to_empty() -> None:
    """Ruri の軸が引けなくても（次元違い・表が無いなど）、想起は bge-m3 だけで続く。"""
    mem = _mem()
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        mem.save_with_id("ruri degrade " + uuid.uuid4().hex, materialize_now=True)
    obs = mem._observations
    assert obs.by_ruri("[1,2]", _BGE_Q, 10) == []
    assert obs.ruri_cosines("[1,2]", ["x"]) == {}
    # 共有の接続が失敗したトランザクションのまま残っていない（巻き戻している）。
    a = "after failure " + uuid.uuid4().hex
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        ida, _ = mem.save_with_id(a, materialize_now=True)
    assert ida and obs.by_ruri(_q(a), _BGE_Q, 5)[0]["id"] == ida


def test_memory_prewarms_ruri() -> None:
    """bge-m3 と同じ切り替えで、Ruri も起動時に先に読み込む（最初の書き込みで読み込みを待たない）。"""
    with (
        patch.dict("os.environ", {"FAMILIAR_EMBEDDING_PREWARM": "1"}),
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_RuriModel, "pre_warm") as warm,
    ):
        ObservationMemory()
    warm.assert_called_once()
    # 切り替えを切ったら読まない（全体テストの既定）。
    with (
        patch.dict("os.environ", {"FAMILIAR_EMBEDDING_PREWARM": "0"}),
        patch.object(_RuriModel, "pre_warm") as warm_off,
    ):
        ObservationMemory()
    warm_off.assert_not_called()
