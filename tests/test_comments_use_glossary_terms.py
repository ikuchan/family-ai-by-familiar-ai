"""コードの語は用語一覧の語で書く（環-ab の C・2026-09-28）。

記憶の store 層を「店」と呼ぶ箇所があった。用語一覧の語は「記憶の箱」（store）で、別の語で呼ぶと、同じものを
指していると読み手が気づけない。
"""

from __future__ import annotations

import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent"


def test_the_store_is_not_called_a_shop():
    hits = [
        f"{p.relative_to(SRC)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "店" in line
    ]
    assert hits == []
