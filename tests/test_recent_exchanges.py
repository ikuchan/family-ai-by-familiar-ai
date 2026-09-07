"""継起をさかのぼって、直近のやりとりを取り出す（段 4）。

会話は関係の連なりをたどる経路である（`設計方針_MI間の関係`）。**上限を持たせない。**
会話はどこかで始まってどこかで終わるので、根で自然に止まる。

載せるのは口に出したものだけである。`版` と `見た` は内部の作業記録で、会話ではない。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.relations import RelationStore


@pytest.fixture
def ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


def _write(ctx, content: str, *, when: datetime) -> str:
    oid = str(uuid.uuid4())
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind) "
                "VALUES (%s, %s, %s, '発話', 'observation')",
                (oid, content, when),
            )
        conn.commit()
    return oid


def _turn(ctx, store, n: int, prev: "str | None", when: datetime) -> str:
    """一つのやりとりを書き、前があれば継起でつなぐ。返すのは起点。"""
    ask = _write(ctx, f"問い{n}", when=when)
    ver = _write(ctx, f"版{n}", when=when)
    ans = _write(ctx, f"答え{n}", when=when)
    store.add("やりとり", [(ask, "起点", 0), (ver, "版", 1), (ans, "答え", 2)])
    if prev:
        store.add("継起", [(prev, "前", 0), (ask, "後", 1)])
    return ask


def test_the_walk_goes_back_to_the_root(ctx):
    store = RelationStore(ctx)
    now = datetime.now(timezone.utc)
    a = _turn(ctx, store, 1, None, now - timedelta(minutes=3))
    b = _turn(ctx, store, 2, a, now - timedelta(minutes=2))
    c = _turn(ctx, store, 3, b, now - timedelta(minutes=1))

    got = store.recent_exchanges(c)
    # 古い順に3つ。上限を渡していないのに根まで戻る。
    assert [m["content"] for m in got if m["role"] == "起点"] == ["問い1", "問い2", "問い3"]


def test_the_working_records_are_left_out(ctx):
    """`版` は会話ではない。混ぜると、調べている途中の文字列が履歴として読まれる。"""
    store = RelationStore(ctx)
    now = datetime.now(timezone.utc)
    a = _turn(ctx, store, 1, None, now)
    got = store.recent_exchanges(a)
    assert [m["role"] for m in got] == ["起点", "答え"]


def test_a_root_on_its_own_returns_just_itself(ctx):
    store = RelationStore(ctx)
    a = _turn(ctx, store, 1, None, datetime.now(timezone.utc))
    got = store.recent_exchanges(a)
    assert [m["content"] for m in got] == ["問い1", "答え1"]


def test_an_unknown_origin_returns_nothing(ctx):
    assert RelationStore(ctx).recent_exchanges(str(uuid.uuid4())) == []
