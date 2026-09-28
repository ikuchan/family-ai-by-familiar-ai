"""Tests for store/relations.py（MI 間の関係の口・段 1）.

記録どうしの関係を、二項の列から多項の関係へ移す（`設計方針_MI間の関係` v0.1）。
段 1 は器と口だけを置く。既存の経路からは呼ばないので挙動は変わらない。

ここで確かめるのは、書いたものが**順序どおり残ること**である（読む口 `members_of`・`relations_for` は
呼び手が無く環-ab で外したので、読み返しは試験の中の問い合わせで行う）。想起の絞りと連なりの辿りは、使う段（2 と 4）で問い合わせの形が
決まってから足す。
"""

from __future__ import annotations

import uuid

import psycopg2.extras
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


def _members(relation_id: int) -> list[dict]:
    """書いた項を読み返す（試験の道具）。読む口 `members_of` は呼び手が無く環-ab で外した。"""
    db = get_db()
    with db.lock:
        with db.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT obs_id, role, position FROM relation_members "
                "WHERE relation_id = %s ORDER BY position ASC NULLS LAST, obs_id",
                (relation_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def _relations(obs_id: str) -> list[int]:
    """その観測が項として入る関係の id（試験の道具）。読む口 `relations_for` は環-ab で外した。"""
    db = get_db()
    with db.lock:
        with db.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT relation_id FROM relation_members WHERE obs_id = %s ORDER BY relation_id",
                (obs_id,),
            )
            return [int(r["relation_id"]) for r in cur.fetchall()]


def test_store_is_built_from_context_alone() -> None:
    """層は文脈だけで組み立てられる。"""
    assert RelationStore(_ctx()) is not None


def test_members_come_back_in_position_order() -> None:
    """やりとりは順序を持つ。書いた順ではなく位置の昇順で返る。"""
    store = RelationStore(_ctx())
    ask, ver, ans = _oid(), _oid(), _oid()
    rid = store.add("やりとり", [(ans, "答え", 2), (ask, "問い", 0), (ver, "版", 1)])
    assert rid is not None
    got = _members(rid)
    assert [m["obs_id"] for m in got] == [ask, ver, ans]
    assert [m["role"] for m in got] == ["問い", "版", "答え"]


def test_members_can_be_appended_after_the_relation_exists() -> None:
    """既にある関係の末尾へ項を足せる（要約は背景で遅れて来る・2026-09-13）。

    位置を持たせずに渡した項は、いまの最大位置の次に置かれる。
    """
    store = RelationStore(_ctx())
    ask, ans, conv = _oid(), _oid(), _oid()
    rid = store.add("やりとり", [(ask, "起点", 0), (ans, "答え", 1)])
    assert rid is not None
    store.extend(rid, [(conv, "要約", None)])
    got = _members(rid)
    assert [(m["obs_id"], m["role"], m["position"]) for m in got] == [
        (ask, "起点", 0),
        (ans, "答え", 1),
        (conv, "要約", 2),
    ]


def test_a_relation_without_order_keeps_every_member() -> None:
    """共起に前後は無い。位置を空けても項は全部返る。"""
    store = RelationStore(_ctx())
    ids = [_oid() for _ in range(3)]
    rid = store.add("共起", [(o, "項", None) for o in ids])
    assert rid is not None
    assert {m["obs_id"] for m in _members(rid)} == set(ids)
    assert all(m["position"] is None for m in _members(rid))


def test_a_relation_with_no_members_is_not_written() -> None:
    """項の無い関係は関係ではない。`save_wr` と同じ約束にする。"""
    store = RelationStore(_ctx())
    assert store.add("やりとり", []) is None


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
    assert _relations(obs) == []
