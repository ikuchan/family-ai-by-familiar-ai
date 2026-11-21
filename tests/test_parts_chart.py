"""環-bb：部品どうしのつながりの図を、コードから作る（2026-11-21 本人「部品の図も作る」「設計の部品で」）。

`parts_chart.mermaid()` が、ソースの import を読み、設計図の部品（TIF・GMD・B・AIF・DIF・CORE・RES・W・O・C・SS と、実物に
ある内省 REST・画面 UI・起動と基盤 BOOT）どうしの矢印にする。どのファイルがどの部品かは、`parts_chart` の対応表で決める。
設計図の文書（`設計図_Mermaid` の「①-2」）の図は手で書かず、これを貼る。
"""

from __future__ import annotations

import re
from pathlib import Path

from familiar_agent import parts_chart as pc

DOCS = Path(__file__).resolve().parents[1] / "docs" / "NewModesDocs"


def test_every_source_file_belongs_to_a_part():
    assert pc.unassigned() == []


def test_imports_become_arrows_between_parts():
    edges = pc.edges()
    assert edges[("CORE", "RES")] >= 1  # メインは判定（Jev）を呼ぶ
    assert edges[("DIF", "SS")] >= 1  # 機器 IF は状態図へ事象を送る
    assert all(a != b for a, b in edges)  # 同じ部品の中の import は描かない


def test_a_relative_module_import_is_resolved():
    """`from ..store import clock` は、モジュール clock（O の部品）への import として数える。"""
    assert "store/clock" in pc.imported_modules("core/self_image")


def test_the_design_document_shows_the_parts_chart_made_from_the_code():
    [doc] = list(DOCS.glob("familiar-ai_設計図_Mermaid_v*.md"))
    text = doc.read_text(encoding="utf-8")
    sec = text[text.index("## ①-2 コードから作った部品の図") :]
    m = re.search(r"```mermaid\n(.*?)```", sec, re.S)
    assert m and m.group(1) == pc.mermaid(), (
        "①-2 の図がコードとずれている。`uv run python -m familiar_agent.parts_chart` の出力を貼る"
    )
