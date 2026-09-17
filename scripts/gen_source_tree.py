"""`docs/ソースツリー.md` を実物から作り直す。

各 module の docstring の 1 行目を説明に使う。`__init__.py`・`__pycache__`・`locales/` は
載せない。使い方：`uv run python scripts/gen_source_tree.py > docs/ソースツリー.md`
"""

from __future__ import annotations

import ast
import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "familiar_agent"
WIDTH = 30  # 名前の欄の幅（`# 説明` の縦を揃える）
DESC_MAX = 64  # 説明の 1 行目をここで切る


def _doc_first_line(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree, clean=True) or ""
    line = doc.strip().split("\n", 1)[0].strip()
    return line[:DESC_MAX]


def _entries(d: Path) -> list[Path]:
    files = sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
    )
    dirs = sorted(
        p
        for p in d.iterdir()
        if p.is_dir() and p.name not in ("__pycache__", "locales") and any(p.rglob("*.py"))
    )
    return files + dirs


def _render(d: Path, prefix: str, out: list[str]) -> None:
    items = _entries(d)
    for i, p in enumerate(items):
        last = i == len(items) - 1
        branch = "└── " if last else "├── "
        if p.is_dir():
            out.append(f"{prefix}{branch}{p.name}/")
            _render(p, prefix + ("    " if last else "│   "), out)
        else:
            name = p.name.ljust(WIDTH - len(branch) + 4)
            desc = _doc_first_line(p)
            out.append(f"{prefix}{branch}{name}# {desc}".rstrip())


def main() -> None:
    lines: list[str] = []
    _render(ROOT, "", lines)
    today = _dt.date.today().isoformat()
    n = sum(
        1 for _ in ROOT.rglob("*.py") if _.name != "__init__.py" and "__pycache__" not in _.parts
    )
    print("# familiar-ai ソースツリー")
    print()
    print(
        "`src/familiar_agent/` のモジュール構成。**この一覧は実物から生成している**——各 module の"
    )
    print("docstring の1行目を説明に使うので、file が増減すれば作り直すだけでよい（説明そのものは")
    print("コードの側にある）。最新はリポジトリを直接見る。")
    print()
    print(
        f"（{today} 生成・`uv run python scripts/gen_source_tree.py > docs/ソースツリー.md`・{n} module）"
    )
    print()
    print("```text")
    print("src/familiar_agent/")
    print("\n".join(lines))
    print("```")
    print()
    print("`locales/` は翻訳の資源（`.json`）で、module ではないので載せていない。")


if __name__ == "__main__":
    main()
