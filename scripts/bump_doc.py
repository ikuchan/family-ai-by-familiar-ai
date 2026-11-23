#!/usr/bin/env python3
"""設計文書の版を 1 つ上げる（2026-11-23）。ファイル名の版・題の（vX.Y）・更新履歴の先頭の 1 行・索引（README）のリンクを直す。

    uv run python scripts/bump_doc.py docs/NewModesDocs/familiar-ai_課題8_段取り設計_v0_679.md "〇〇を済みにした（2026-11-23）。"

以前は会話ごとに一時フォルダへ書いていた助けで、セッションが替わると消えた。文書の検査
（`tests/test_docs_are_consistent.py`）が、題の版＝ファイル名の版、更新履歴にその版の項、索引のリンクを確かめる。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_VERSION = re.compile(r"_v(\d+)_(\d+)\.md$")


def bump(path: Path, note: str) -> Path:
    """`path` の版を 1 つ上げた新しいファイルを書き、古いファイルを消し、同じフォルダの README.md のリンクを直す。"""
    m = _VERSION.search(path.name)
    if m is None:
        raise ValueError(f"ファイル名に版（_vX_Y.md）が無い：{path.name}")
    major, minor = m.group(1), int(m.group(2))
    old_v, new_v = f"{major}.{minor}", f"{major}.{minor + 1}"
    new_path = path.with_name(_VERSION.sub(f"_v{major}_{minor + 1}.md", path.name))
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[0] = lines[0].replace(f"v{old_v}", f"v{new_v}", 1)
    text = "\n".join(lines)
    head = text.find("## 更新履歴")
    if head < 0:
        raise ValueError(f"「## 更新履歴」が無い：{path.name}")
    at = text.find("\n", head) + 1
    while at < len(text) and text[at] == "\n":
        at += 1
    text = text[:at] + f"> v{new_v}：{note}\n>\n" + text[at:]
    new_path.write_text(text, encoding="utf-8")
    path.unlink()
    index = path.parent / "README.md"
    if index.exists():
        index.write_text(
            index.read_text(encoding="utf-8").replace(path.name, new_path.name), encoding="utf-8"
        )
    return new_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="版を上げる文書（docs/NewModesDocs/…_vX_Y.md）")
    ap.add_argument("note", help="更新履歴に足す 1 行（日付つき）")
    args = ap.parse_args()
    print(bump(args.path, args.note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
