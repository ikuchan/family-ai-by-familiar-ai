"""過去の「つなぎ」の記録は、やりとりの項としては残り、想起には出さない（段 4）。

つなぎを言う仕組みは撤去した（出-cl 段 3・2026-10-15）が、O には役割 `つなぎ` の記録が残っている。想起に出さない
役割は二つある。`旧`（もう現行ではない）と `つなぎ`（口に出したが、覚えておく中身がない）である。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext, viewpoint_of
from familiar_agent.store.observations import ObservationStore
from familiar_agent.store.relations import RelationStore
from familiar_agent.store.situated import SituatedVectors

_VEC = "[" + ",".join(["0.03125"] * 1024) + "]"


@pytest.fixture
def ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


def test_a_filler_does_not_come_back_from_recall(ctx):
    """つなぎは想起に出ない。出ると、中身の無い一言が候補を食う（054 の理由）。"""
    store = ObservationStore(ctx, situated=SituatedVectors(ctx))
    mark = f"つなぎの検査 {uuid.uuid4()}"
    oid = str(uuid.uuid4())
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction) "
                "VALUES (%s, %s, %s, '発話')",
                (oid, f"つなぎに言った：{mark}", datetime.now(timezone.utc)),
            )
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, relation_key, vector, content) "
                "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                (str(uuid.uuid4()), oid, viewpoint_of(DEFAULT_PERSON_ID), _VEC, mark),
            )
        conn.commit()

    def _found():
        return {
            str(r.get("memory_id") or r.get("id"))
            for r in store.recency_fallback(400)
            if mark in str(r.get("summary") or r.get("content") or "")
        }

    # 役割を付ける前は出る（絞りが常に偽になっていないことの確認）。
    assert _found() == {oid}
    RelationStore(ctx).add("やりとり", [(oid, "つなぎ", 0)])
    assert _found() == set()
