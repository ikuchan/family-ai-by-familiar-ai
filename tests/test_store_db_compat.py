"""Tests for store/db_compat.py（接続ラッパの移動）.

PostgreSQL の RealDict 接続と、テストで使う sqlite3 接続の差を吸収する薄い
ラッパ。SQL も想起ロジックも持たない純粋なインフラなので、`store/` へ移す。
挙動は変えない。
"""

from __future__ import annotations

from familiar_agent.store.db_compat import (
    _RealDictConnWrapper,
    _SQLiteConnWrapper,
    _SQLiteCursorWrapper,
)


def test_wrappers_are_importable_from_store() -> None:
    for cls in (_RealDictConnWrapper, _SQLiteConnWrapper, _SQLiteCursorWrapper):
        assert isinstance(cls, type)


def test_memory_module_uses_the_moved_wrappers() -> None:
    """memory.py 側が同じ実体を指している（二重定義になっていない）。"""
    from familiar_agent.tools import memory as memory_module

    assert memory_module._RealDictConnWrapper is _RealDictConnWrapper
    assert memory_module._SQLiteConnWrapper is _SQLiteConnWrapper
