"""git の一覧を読む番人は、まだ git に入れていない新しいファイルも見る（2026-11-22）。

`run_tests.sh` は、試験が緑になってから `git add -A` する。試験のあいだ新しいファイルは git に無く、`git ls-files` だけで
集める番人（家族の名前・部品の図）は、それを見ずに緑になっていた（記-ae 段 1 の試験の作り名・段 2 の部品の図）。
見るのは `git add -A` が入れるものと同じ（追跡しているもの＋無視していない新しいもの）。
"""

from __future__ import annotations

import pathlib
import subprocess


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "kept.py", ".gitignore"], check=True)
    (tmp_path / "new.py").write_text("y = 2\n", encoding="utf-8")  # まだ git に入れていない
    (tmp_path / "ignored.py").write_text("z = 3\n", encoding="utf-8")  # 無視するもの
    return tmp_path


def test_the_git_file_list_has_new_files_but_not_ignored_or_deleted_ones(tmp_path):
    """家族の名前の番人と部品の図は、どちらもこの一覧を読む。"""
    from familiar_agent.parts_chart import git_files

    root = _repo(tmp_path)
    (root / "gone.py").write_text("w = 4\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "gone.py"], check=True)
    (root / "gone.py").unlink()  # git rm でなく消した（追跡したまま）
    got = set(git_files(root))
    assert {"kept.py", "new.py"} <= got and not {"ignored.py", "gone.py"} & got


def test_the_parts_chart_sees_new_files(tmp_path, monkeypatch):
    from familiar_agent import parts_chart

    root = _repo(tmp_path)
    monkeypatch.setattr(parts_chart, "_SRC", root)
    caches = (parts_chart._modules, parts_chart.imported_modules, parts_chart.edges)
    for c in caches:
        c.cache_clear()
    try:
        got = set(parts_chart._modules())
    finally:
        for c in caches:
            c.cache_clear()
    assert {"kept", "new"} <= got and "ignored" not in got
