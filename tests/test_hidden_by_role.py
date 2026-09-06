"""役割「旧」を持つ記録が想起から外れることを確かめる（段 2）。

`superseded_by` 列を落とし、**役割 `旧` が「もう現行ではない」を表す**ことにした
（`設計方針_MI間の関係` v0.3）。種類（改訂、畳み込み、解決、前進）は理由を言うだけで、
隠すかどうかは役割だけで決まる。

**片側だけでは確かめたことにならない。** 隠した記録が消えることと、隠していない記録が
残ることを、同じ経路で対にして見る。絞りが常に真や常に偽になっていれば、どちらかが落ちる。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from familiar_agent.legacy.semantic_layer import LegacySemanticLayer
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext, viewpoint_of
from familiar_agent.store.observations import ObservationStore
from familiar_agent.store.relations import RelationStore
from familiar_agent.store.situated import SituatedVectors
from familiar_agent.db import get_db

_VEC = "[" + ",".join(["0.03125"] * 1024) + "]"


@pytest.fixture
def ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


@pytest.fixture
def store(ctx) -> ObservationStore:
    return ObservationStore(ctx, situated=SituatedVectors(ctx), legacy=LegacySemanticLayer(ctx))


def _write_pair(ctx) -> tuple[str, str, str]:
    """同じ言葉・同じ面で二つ書き、片方だけを隠す。返すのは (隠す, 残す, 目印)。"""
    mark = f"役割の検査 {uuid.uuid4()}"
    ids = []
    now = datetime.now(timezone.utc)
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            for i in range(2):
                oid = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO observations (id, content, timestamp, direction, kind) "
                    "VALUES (%s, %s, %s, '発話', 'observation')",
                    (oid, mark, now - timedelta(seconds=i)),
                )
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, relation_key, vector, content) "
                    "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                    (str(uuid.uuid4()), oid, viewpoint_of(DEFAULT_PERSON_ID), _VEC, mark),
                )
                ids.append(oid)
        conn.commit()
    RelationStore(ctx).add("改訂", [(ids[0], "旧", 0), (str(uuid.uuid4()), "新", 1)])
    return ids[0], ids[1], mark


def _found(rows, mark: str) -> set[str]:
    """目印を含む行の観測 id を集める。

    層によって返り値の鍵が違う（素の行は `id`/`content`、想起の形に整えたものは
    `memory_id`/`summary`）。どちらでも拾えるようにしておかないと、絞りが効いている
    のか行の読み方を間違えているのかが区別できない。
    """
    got = set()
    for r in rows:
        text = str(r.get("content") or r.get("summary") or "")
        oid = r.get("id") or r.get("memory_id")
        if oid and mark in text:
            got.add(str(oid))
    return got


def test_the_vector_recall_skips_the_old_one(ctx, store):
    hidden, kept, mark = _write_pair(ctx)
    got = _found(store.by_vector(_VEC, 400), mark)
    assert kept in got
    assert hidden not in got


def test_the_situated_read_skips_the_old_one(ctx, store):
    hidden, kept, mark = _write_pair(ctx)
    rows = store._read_observations_by_situated(
        viewpoint_of(DEFAULT_PERSON_ID),
        400,
        ("id", "content"),
    )
    got = _found(rows, mark)
    assert kept in got
    assert hidden not in got


def test_the_recency_fallback_skips_the_old_one(ctx, store):
    hidden, kept, mark = _write_pair(ctx)
    got = _found(store.recency_fallback(400, None), mark)
    assert kept in got
    assert hidden not in got


def test_the_keyword_fallback_skips_the_old_one(ctx, store):
    hidden, kept, mark = _write_pair(ctx)
    got = _found(store.keyword_fallback(mark, 400, None), mark)
    assert kept in got
    assert hidden not in got


def test_any_kind_of_relation_hides_when_the_role_is_old(ctx, store):
    """隠すのは役割であって種類ではない。畳み込みでも解決でも同じに隠れる。"""
    for kind in ("畳み込み", "解決", "前進", "未分類"):
        mark = f"種類の検査 {kind} {uuid.uuid4()}"
        oid = str(uuid.uuid4())
        with ctx.lock:
            conn = ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO observations (id, content, timestamp, direction, kind) "
                    "VALUES (%s, %s, %s, '発話', 'observation')",
                    (oid, mark, datetime.now(timezone.utc)),
                )
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, relation_key, vector, content) "
                    "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                    (str(uuid.uuid4()), oid, viewpoint_of(DEFAULT_PERSON_ID), _VEC, mark),
                )
            conn.commit()
        assert _found(store.recency_fallback(400, None), mark) == {oid}, kind
        RelationStore(ctx).add(kind, [(oid, "旧", 0), (str(uuid.uuid4()), "新", 1)])
        assert _found(store.recency_fallback(400, None), mark) == set(), kind


def test_a_member_in_another_role_is_not_hidden(ctx, store):
    """`新` や `項` で入っただけの記録は隠れない。役割を見ずに存在だけを見ていたら落ちる。"""
    mark = f"役割違いの検査 {uuid.uuid4()}"
    oid = str(uuid.uuid4())
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind) "
                "VALUES (%s, %s, %s, '発話', 'observation')",
                (oid, mark, datetime.now(timezone.utc)),
            )
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, relation_key, vector, content) "
                "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                (str(uuid.uuid4()), oid, viewpoint_of(DEFAULT_PERSON_ID), _VEC, mark),
            )
        conn.commit()
    RelationStore(ctx).add("改訂", [(str(uuid.uuid4()), "旧", 0), (oid, "新", 1)])
    RelationStore(ctx).add("共起", [(oid, "項", None), (str(uuid.uuid4()), "項", None)])
    assert _found(store.recency_fallback(400, None), mark) == {oid}
