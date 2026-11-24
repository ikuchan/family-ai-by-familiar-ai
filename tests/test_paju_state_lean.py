"""環-bb：パジュの状態図を Lean 4 の模型にして、決まりを証明する（2026-11-21 本人の承認「リーンは入れていいです」）。

`loop/paju_state_lean.lean()` が、状態図の型から `lean/PajuState.lean` を作る。置いたファイルが作ったものと違えば落ちる（模型が
コードからずれない）。Lean（`~/.elan/bin/lean`）がある機械では、そのファイルの証明が通ることも確かめる。決まりは仮（本人の確認待ち）：
行き止まりが無い、寝ているあいだ耳は言葉を頭へ渡さない。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from familiar_agent.loop import paju_state_lean

ROOT = Path(__file__).resolve().parents[1]
LEAN_FILE = ROOT / "lean" / "PajuState.lean"


def _lean() -> "str | None":
    found = shutil.which("lean")
    home = Path.home() / ".elan" / "bin" / "lean"
    return found or (str(home) if home.exists() else None)


def test_the_lean_model_is_the_one_made_from_the_code():
    assert LEAN_FILE.read_text(encoding="utf-8") == paju_state_lean.lean(), (
        "lean/PajuState.lean がコードとずれている。"
        "`uv run python -m familiar_agent.loop.paju_state_lean > lean/PajuState.lean` で作り直す"
    )


def test_the_toolchain_is_pinned():
    assert (ROOT / "lean" / "lean-toolchain").read_text().strip() == "leanprover/lean4:v4.15.0"


def test_the_model_states_the_rules():
    text = paju_state_lean.lean()
    assert "theorem sleeping_takes_no_request" in text
    # 2026-11-21 本人が足した 2 つ（違う問いが同時に待たない・黙っているあいだ話さない）
    assert "theorem a_new_question_replaces_the_old" in text
    assert "theorem silenced_takes_no_outside_voice" in text
    for region in (
        "head",
        "listening",
        "window",
        "awaited",
        "entrance",
        "hush",
        "voice",
        "alarm",
        "song",
    ):
        assert f"theorem {region}_has_no_dead_end" in text, region


@pytest.mark.skipif(_lean() is None, reason="Lean が無い機械では証明を回さない")
def test_the_rules_are_proved_by_lean():
    got = subprocess.run([_lean(), str(LEAN_FILE)], capture_output=True, text=True, timeout=300)
    assert got.returncode == 0, got.stdout[-2000:] + got.stderr[-2000:]
