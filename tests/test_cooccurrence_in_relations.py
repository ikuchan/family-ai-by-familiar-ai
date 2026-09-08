"""共起を関係の器で書き、引く（段 5）。

拡散想起は「過去に一緒に想起された記録」をたどる。その母集合が共起の関係である。

**種類で絞らないと、やりとりや改訂が共起として数えられる。** 一つの表に全部の関係が
載っているので、絞りを落とすと「同じターンの問いと答え」が共起として返ってくる。
"""

from __future__ import annotations

import uuid

import pytest

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.relations import RelationStore, combine_cooccurring_ids


@pytest.fixture
def store() -> RelationStore:
    db = get_db()
    return RelationStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )


def _obs(store: RelationStore, n: int) -> list:
    """観測を n 件書いて id を返す。

    共起の読み出しは `observations` と結合する（自己認識の記録を除くため）。実体の無い
    id では返ってこないので、行を作ってから引く。
    """
    ids = []
    with store._ctx.lock:
        conn = store._ctx.conn()
        with conn.cursor() as cur:
            for i in range(n):
                oid = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO observations (id, content, timestamp, direction, kind) "
                    "VALUES (%s, %s, now(), '発話', 'observation')",
                    (oid, f"共起の検査 {oid}"),
                )
                ids.append(oid)
        conn.commit()
    return ids


def test_a_cooccurrence_is_written_and_found(store):
    a, b, c = _obs(store, 3)
    store.record_cooccurrence([a, b, c])
    # a と b を含む過去の集まりから、残りの c が返る。
    assert store.cooccurring([a, b], min_shared=2, limit=10) == [c]


def test_the_ids_asked_for_are_not_returned(store):
    a, b, c = _obs(store, 3)
    store.record_cooccurrence([a, b, c])
    assert a not in store.cooccurring([a, b], min_shared=2, limit=10)


def test_one_shared_item_is_not_enough(store):
    a, b, c = _obs(store, 3)
    store.record_cooccurrence([a, b, c])
    assert store.cooccurring([a], min_shared=2, limit=10) == []


def test_an_exchange_is_not_counted_as_cooccurrence(store):
    """**反証側。** 種類で絞っていなければ、やりとりの項が共起として返る。"""
    a, b, c = _obs(store, 3)
    store.add("やりとり", [(a, "起点", 0), (b, "答え", 1), (c, "要約", 2)])
    assert store.cooccurring([a, b], min_shared=2, limit=10) == []


def test_an_empty_group_is_not_written(store):
    assert store.record_cooccurrence([]) is None


def test_combine_keeps_order_and_drops_duplicates():
    """WR に入れる id は、W の想起 MI ＋ そのターンの新記憶。順序保存で重複除去。"""
    memories = [{"memory_id": "w1"}, {"memory_id": "w2"}, {"memory_id": "w1"}]
    assert combine_cooccurring_ids(memories, [None, "obs1", "w1"]) == ["w1", "w2", "obs1"]
    assert combine_cooccurring_ids(None, None) == []
