"""Embodied agent - a real-world exploration AI."""
from __future__ import annotations

import subprocess
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent


def _compute_version() -> str:
    try:
        hash_r = subprocess.run(
            ["git", "rev-parse", "--short=5", "HEAD"],
            capture_output=True, text=True, cwd=_SRC_DIR,
        )
        dirty_r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, cwd=_SRC_DIR,
        )
        if hash_r.returncode == 0 and hash_r.stdout.strip():
            dirty = bool(dirty_r.stdout.strip())
            return f"v0.{hash_r.stdout.strip()}{'*' if dirty else ''}"
    except Exception:
        pass
    return "v0.?????"


__version__ = _compute_version()
