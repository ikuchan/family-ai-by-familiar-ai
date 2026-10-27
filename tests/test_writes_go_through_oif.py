"""記録を書く口は OIF だけ（記-u 段 3・2026-10-27・本人の決定）。

REST の 4 か所（内省パス・季節・設定値・自己像）は、OIF を飛ばして記憶の口（`save_async_with_id`）を直接呼び、向き「内省」に
種類 `observation` を渡していた。向きから種類を決める OIF の表（「内省」→ `self_model`）と食い違っていたのは、この 4 か所
だけだった。書く口を OIF に一本化し、記憶の口を呼ぶのは OIF の中だけにする。呼び手の無い同期版 `save` は外した。
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "familiar_agent"


def _callers(names: "set[str]") -> "list[str]":
    """本体の中で、`names` の口を呼んでいるファイル（AST で呼び出しだけを数える。文中の言及は数えない）。"""
    out = set()
    for p in _SRC.rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if any(
            isinstance(n, ast.Call) and getattr(n.func, "attr", None) in names
            for n in ast.walk(tree)
        ):
            out.add(p.relative_to(_SRC).as_posix())
    return sorted(out)


def test_only_the_oif_calls_the_memory_mouth():
    """本体の中で記憶の口を呼ぶのは OIF だけ（記憶の口の非同期版は、同期版を呼ばずにスレッドへ渡す）。"""
    assert _callers({"save_async_with_id", "save_with_id"}) == ["io/oif.py"]


def test_the_scan_finds_a_caller():
    """**反証の側**：走査が空振りしていないこと。OIF の書く口の呼び手は REST にもある。"""
    assert "loop/rest.py" in _callers({"write"})


def test_the_unused_sync_save_is_gone():
    from familiar_agent.tools.memory import ObservationMemory

    assert not hasattr(ObservationMemory, "save")
