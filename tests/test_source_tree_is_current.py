"""`docs/ソースツリー.md` は生成物（`scripts/gen_source_tree.py`）。いまのコードから作った出力と一致していること（2026-11-23）。

10/17 に作ったまま 128 本しか載っておらず、208 本のうち 80 本が抜けていた。部品の図（`test_parts_chart`）と同じく、ずれたら
全体テストで気づく。直し方：`uv run python scripts/gen_source_tree.py > docs/ソースツリー.md`。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_source_tree_doc_is_generated_from_the_code():
    out = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "gen_source_tree.py")],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    doc = (_ROOT / "docs" / "ソースツリー.md").read_text(encoding="utf-8")
    assert doc == out, (
        "docs/ソースツリー.md が古い。`uv run python scripts/gen_source_tree.py > docs/ソースツリー.md`"
    )
