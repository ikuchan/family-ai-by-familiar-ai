"""min_score の是正：合成 final score の床であること（生コサインの床ではない）。

根拠台帳 §3–4 の確定方針では、関連 r は段階化して門にせず、無関係の最終排除は
合成5軸スコアの床＝min_score が担う。ここでは (1) 床が合成スコアに効く（生コサイン
ではない）、(2) 床を課すときは候補を過剰取得する、(3) store は素取得だけを持つ、を見る。
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from unittest.mock import patch

import psycopg2
import pytest

from familiar_agent.store import clock
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


@pytest.fixture()
def memory():
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


def _age_observation(content: str, days: int) -> None:
    """指定内容の観測を days 日前へ古くし、再想起履歴を消す（t を落とすため）。"""
    old = clock.now_utc() - timedelta(days=days)
    raw = psycopg2.connect(os.environ["DATABASE_URL"])
    raw.autocommit = True
    with raw.cursor() as cur:
        cur.execute(
            "UPDATE observations SET timestamp=%s, last_recalled_at=NULL WHERE content=%s",
            (old, content),
        )
    raw.close()


def test_min_score_floors_on_composite_not_cosine(memory):
    """古い観測は composite<cosine。中間に床を置くと合成床は切り、生コサイン床は残す。"""
    content = f"min_score composite {uuid.uuid4()}"
    memory.save(content, kind="observation", emotion="neutral")
    _age_observation(content, days=30)  # t をほぼ floor まで落とす → M<1

    base = memory.recall(content, n=5, min_score=0.0)
    item = next((r for r in base if r["summary"] == content), None)
    assert item is not None, "床なしで対象が想起されない（前提が崩れている）"

    composite = item["score"]
    cosine = item["confidence"] * 2.0 - 1.0
    assert composite < cosine, f"ギャップが無い（M=1）: composite={composite} cosine={cosine}"

    mid = (composite + cosine) / 2.0  # composite < mid < cosine
    filtered = memory.recall(content, n=5, min_score=mid)
    assert content not in [r["summary"] for r in filtered], (
        "合成スコアが床未満なのに残っている（生コサインで絞っている）"
    )


def test_overfetch_only_when_min_score_positive(memory):
    """min_score>0 のとき候補を n*3（上限20）取り、0 のとき n ちょうど取る。"""
    with (
        patch.object(memory._observations, "by_vector", return_value=[]) as bv,
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        memory.recall("q", n=5, min_score=0.05)
        assert bv.call_args.args[1] == 15, "min_score>0 で n*3 を取っていない"

        bv.reset_mock()
        memory.recall("q", n=10, min_score=0.05)
        assert bv.call_args.args[1] == 20, "上限20 で頭打ちになっていない"

        bv.reset_mock()
        memory.recall("q", n=5, min_score=0.0)
        assert bv.call_args.args[1] == 5, "min_score=0 で n ちょうどを取っていない"


def test_by_vector_has_no_min_cosine():
    """store の by_vector は素取得だけ（合成床の引数を持たない）。"""
    import inspect

    from familiar_agent.store.observations import ObservationStore

    sig = inspect.signature(ObservationStore.by_vector)
    assert "min_cosine" not in sig.parameters, "by_vector に min_cosine が残っている"
