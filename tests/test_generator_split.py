"""反復ごとの材料を組む部分を `loop/generator.py` へ出す（環-e-に 段1）。

**状態を1つも触らない部分から出す。** `_present_ctx` と `_pi_ctx` は
`InformationProcessing` の可変状態（38 個）をどれも読まず、引数と外の登録簿だけで
文字列を組む。純粋な核から出せば、切り方が正しいかを危険の小さいところで確かめられる
（`モジュール分割設計` の環-e-に）。

**挙動は変えない。** 中身も名前も動かさず、置き場所だけを移す。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from familiar_agent.loop import event_loop, generator

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def test_the_context_builders_live_in_the_generator():
    assert callable(generator._present_ctx)
    assert callable(generator._pi_ctx)


def test_the_loop_no_longer_defines_them():
    """移し終わったことを、定義が残っていないことで見る。"""
    tree = ast.parse(_LOOP.read_text(encoding="utf-8"))
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_present_ctx" not in names
    assert "_pi_ctx" not in names


def test_the_loop_still_reaches_them():
    """置き場所が変わっても、ループからは同じ名前で見える（挙動を変えない）。"""
    assert event_loop._present_ctx is generator._present_ctx
    assert event_loop._pi_ctx is generator._pi_ctx


def test_they_do_not_touch_the_loop_state():
    """出した先がループの状態を読まないこと。読めば、切り方が誤っている。"""
    for fn in (generator._present_ctx, generator._pi_ctx):
        src = inspect.getsource(fn)
        assert "self." not in src
        assert "InformationProcessing" not in src
