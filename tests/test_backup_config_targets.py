"""git の外にある、失うと戻らないファイルを毎晩 Google Drive へ送る（`scripts/backup_config.sh`・2026-10-05）。

2026-08 に `ME.md` と `FAMILY.md` を古いディスクごと失った（gitignore されていて機外の控えが無かった）。退避の一覧は
人が書く設定ファイルが増えるたびに足さないと、黙って漏れる。2026-10-05 に `MUSIC.md`（音楽の名前の表）と
`PEOPLE.md`（家族以外の人）が漏れていたので足し、読む仕組みごと撤去した `ROUTINES.md` を外した。
`~/.familiar_ai/` の小さな 2 つ（Spotify の鍵・作業メモ）も送る。撮った写真（`captures/`）は大きいので送らない。
"""

from __future__ import annotations

import pathlib
import re

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "backup_config.sh"


def _array(name: str) -> list[str]:
    m = re.search(rf"^{name}=\(([^)]*)\)", SCRIPT.read_text(encoding="utf-8"), re.M)
    assert m, name
    return m.group(1).split()


def test_the_repo_files_cover_every_hand_written_setting():
    assert set(_array("REPO_FILES")) == {".env", "ME.md", "FAMILY.md", "MUSIC.md"}
    assert set(_array("OPTIONAL_REPO_FILES")) == {"PEOPLE.md"}  # 無くても「missing」と言わない


def test_the_small_files_in_the_home_folder_are_sent():
    assert set(_array("HOME_FILES")) == {"spotify_token.json", "FAMILY_答え_作業中.md"}


def test_every_gitignored_setting_is_listed():
    """gitignore された人の設定（`*.md` と `.env`）は、どれかの一覧に入っている。"""
    gitignore = (SCRIPT.parents[1] / ".gitignore").read_text(encoding="utf-8").split()
    hand_written = {g for g in gitignore if g.endswith(".md") and "/" not in g and "*" not in g}
    listed = set(_array("REPO_FILES")) | set(_array("OPTIONAL_REPO_FILES"))
    assert hand_written - {"ROUTINES.md"} <= listed, hand_written - listed
