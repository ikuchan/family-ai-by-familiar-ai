"""実在しない module を import していないこと。

`core/brief_turn.py`（のちに到達不能として撤去）が `TYPE_CHECKING` の中で `from ..social_policy import
SocialPolicyDecision` を書いていた。**その module は存在しない。** 実行時には評価されない
ので落ちず、mypy も追えないまま通っていた。型が実体を指していない状態である。

数え上げでは網羅を証明できないので、**`src/` 全部の相対 import を引いて実在を確かめる**。
撤去のたびに取り残しが出ないようにする（`capabilities.yaml` の `_KEY_MODULES` が
6件も痩せていたのと同じ性質・出-j）。
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src/familiar_agent"


def _target_exists(base: Path, module: str) -> bool:
    """相対 import の行き先が、module（.py）か package（ディレクトリ）として在るか。"""
    p = base
    for part in module.split("."):
        p = p / part
    return p.with_suffix(".py").exists() or (p / "__init__.py").exists()


def test_every_relative_import_points_at_something_that_exists():
    missing: list[str] = []
    for f in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
                continue
            # level=1 は自分のディレクトリ、level=2 はその親。
            base = f.parent
            for _ in range(node.level - 1):
                base = base.parent
            if not _target_exists(base, node.module):
                dots = "." * node.level
                missing.append(f"{f.relative_to(_SRC)}: from {dots}{node.module} import ...")
    assert missing == [], "実在しない module を import している:\n" + "\n".join(missing)
