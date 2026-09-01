"""Tests for store/persons.py（人物レジストリの切り出し・S6d）.

`persons` は人物レジストリ。視点ベクトル（`perspective_vec`）は 045 で落とした。
ストアの一部だが、観測とは別の表なので持ち主を分ける。

外からは `person_memory_manager.py` が `ObservationMemory` 越しに呼ぶので、
公開面の委譲は残す。挙動は変えない。
"""

from __future__ import annotations

import pathlib
import re
import uuid

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.persons import PersonRegistry
from familiar_agent.tools.memory import ObservationMemory


def _ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


def test_registry_is_built_from_context_alone() -> None:
    """層は文脈だけで組み立てられる。"""
    assert PersonRegistry(_ctx()) is not None


def test_registering_a_person_is_idempotent() -> None:
    """同じ名前を二度登録しても増えず、同じ id が返る。"""
    registry = PersonRegistry(_ctx())
    name = f"unit person {uuid.uuid4()}"
    first = registry.register_person(name)
    second = registry.register_person(name)
    assert first == second, "同じ名前で別の id が振られている"


def test_registered_person_appears_in_the_list() -> None:
    registry = PersonRegistry(_ctx())
    name = f"unit person {uuid.uuid4()}"
    pid = registry.register_person(name)
    assert any(p["id"] == pid for p in registry.list_persons())


def test_public_entry_points_are_still_reachable() -> None:
    """person_memory_manager.py が呼ぶ入口が残っている。"""
    for name in ("register_person", "list_persons"):
        assert hasattr(ObservationMemory, name), name


def test_memory_module_no_longer_owns_the_persons_table() -> None:
    """`persons` の SQL が memory.py に残っていない。

    import とコメントで語は出るので、SQL 文の中だけを見る。
    """
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for stmt in re.findall(r'"[^"]*\b(?:FROM|INTO|UPDATE)\s+persons\b[^"]*"', src):
        raise AssertionError(f"persons の SQL が memory.py に残っている: {stmt}")
