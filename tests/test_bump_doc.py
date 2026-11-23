"""設計文書の版上げの助け（`scripts/bump_doc.py`・2026-11-23）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "bump_doc", Path(__file__).resolve().parents[1] / "scripts" / "bump_doc.py"
)
bump_doc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bump_doc)


def test_the_version_title_history_and_index_move_together(tmp_path):
    doc = tmp_path / "familiar-ai_設計方針_試し_v0_9.md"
    doc.write_text(
        "# 設計方針：試し（v0.9）\n\n本文\n\n## 更新履歴\n\n> v0.9：前の版。\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "- [試し](familiar-ai_設計方針_試し_v0_9.md)\n", encoding="utf-8"
    )
    new = bump_doc.bump(doc, "直した（2026-11-23）。")
    assert new.name == "familiar-ai_設計方針_試し_v0_10.md" and not doc.exists()
    text = new.read_text(encoding="utf-8")
    assert text.startswith("# 設計方針：試し（v0.10）")
    assert "## 更新履歴\n\n> v0.10：直した（2026-11-23）。\n>\n> v0.9：前の版。" in text
    assert "familiar-ai_設計方針_試し_v0_10.md" in (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )
