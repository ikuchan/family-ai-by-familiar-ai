"""Embodied agent - a real-world exploration AI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: 起動のときに渡した DB の向き先（環-an）。パッケージの入口なので、どこから始めても `.env` を読む前に控えられる。
#: `.env` は `override=True` で読むので、`DATABASE_URL=… ./run-gui.sh` と渡しても `.env` の値が勝ち、テスト DB に
#: 向けたつもりが本番 DB を見ていた。`bootstrap` と `env_reload` がファイルを読んだあとで戻す（`keep_launch_database_url`）。
#: 空の値は、渡していないものとして扱う。`--db-url` で渡した値もここに入れる（`main`）。
LAUNCH_DATABASE_URL: str | None = os.environ.get("DATABASE_URL") or None

_SRC_DIR = Path(__file__).resolve().parent


def _compute_version() -> str:
    try:
        hash_r = subprocess.run(
            ["git", "rev-parse", "--short=5", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_SRC_DIR,
        )
        dirty_r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=_SRC_DIR,
        )
        if hash_r.returncode == 0 and hash_r.stdout.strip():
            dirty = bool(dirty_r.stdout.strip())
            return f"v0.{hash_r.stdout.strip()}{'*' if dirty else ''}"
    except Exception:
        pass
    return "v0.?????"


__version__ = _compute_version()
