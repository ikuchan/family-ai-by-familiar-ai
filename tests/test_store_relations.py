"""Tests for store/relations.py（MI 間の関係の口・段 1）.

記録どうしの関係を、二項の列から多項の関係へ移す（`設計方針_MI間の関係` v0.1）。
段 1 は器と口だけを置く。既存の経路からは呼ばないので挙動は変わらない。

ここで確かめるのは、書いたものが**順序どおり読めること**と、**観測から関係を引ける
こと**の二つに尽きる。想起の絞りと連なりの辿りは、使う段（2 と 4）で問い合わせの形が
決まってから足す。
"""

from __future__ import annotations

import uuid

import pytest
from psycopg2.errors import UniqueViolation

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.relations import RelationStore


def _ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


def _oid() -> str:
    return str(uuid.uuid4())


def test_store_is_built_from_context_alone() -> None:
    """層は文脈だけで組み立てられる。"""
    assert RelationStore(_ctx()) is not None


def test_members_come_back_in_position_order() -> None:
    """やりとりは順序を持つ。書いた順ではなく位置の昇順で返る。"""
    store = RelationStore(_ctx())
    ask, ver, ans = _oid(), _oid(), _oid()
    rid = store.add("やりとり", [(ans, "答え", 2), (ask, "問い", 0), (ver, "版", 1)])
    assert rid is not None
    got = store.members_of(rid)
    assert [m["obs_id"] for m in got] == [ask, ver, ans]
    assert [m["role"] for m in got] == ["問い", "版", "答え"]


def test_a_relation_without_order_keeps_every_member() -> None:
    """共起に前後は無い。位置を空けても項は全部返る。"""
    store = RelationStore(_ctx())
    ids = [_oid() for _ in range(3)]
    rid = store.add("共起", [(o, "項", None) for o in ids])
    assert rid is not None
    assert {m["obs_id"] for m in store.members_of(rid)} == set(ids)
    assert all(m["position"] is None for m in store.members_of(rid))


def test_a_relation_with_no_members_is_not_written() -> None:
    """項の無い関係は関係ではない。`save_wr` と同じ約束にする。"""
    store = RelationStore(_ctx())
    assert store.add("やりとり", []) is None


def test_an_observation_finds_the_relations_it_belongs_to() -> None:
    store = RelationStore(_ctx())
    old, new = _oid(), _oid()
    rid = store.add("改訂", [(old, "旧", 0), (new, "新", 1)])
    assert store.relations_for(old) == [rid]
    assert store.relations_for(new) == [rid]


def test_relations_can_be_narrowed_by_kind_and_role() -> None:
    """段 2 の絞りは「改訂の旧として現れるか」を引く。その道が通ることを見る。"""
    store = RelationStore(_ctx())
    obs = _oid()
    revised = store.add("改訂", [(obs, "旧", 0), (_oid(), "新", 1)])
    store.add("共起", [(obs, "項", None), (_oid(), "項", None)])
    assert store.relations_for(obs, kind="改訂") == [revised]
    assert store.relations_for(obs, kind="改訂", role="旧") == [revised]
    # 同じ観測でも、役割が違えば当たらない。
    assert store.relations_for(obs, kind="改訂", role="新") == []


def test_the_same_member_cannot_be_written_twice() -> None:
    """主キーが効いていることの反証側。効いていなければ二重に入って通ってしまう。"""
    store = RelationStore(_ctx())
    obs = _oid()
    with pytest.raises(UniqueViolation):
        store.add("改訂", [(obs, "旧", 0), (obs, "旧", 1)])


def test_a_failed_write_leaves_no_header_behind() -> None:
    """項の書き込みが落ちたら、ヘッダも残さない。項の無い関係が溜まると、関係の数が
    実際のつながりの数と合わなくなる。"""
    store = RelationStore(_ctx())
    obs = _oid()
    with pytest.raises(UniqueViolation):
        store.add("検査", [(obs, "旧", 0), (obs, "旧", 1)])
    # 接続が使える状態のまま残っていることも、ここで同時に見ている。
    assert store.relations_for(obs) == []
