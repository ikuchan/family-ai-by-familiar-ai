"""Tests for capability_state — manifest loader and AI self-understanding storage."""

from __future__ import annotations

import ast
from pathlib import Path

from familiar_agent.capability_state import (
    load_manifest,
    load_summary,
    save_summary,
)

_TOOLS = Path(__file__).parent.parent / "src/familiar_agent/tools"
_SRC = Path(__file__).parent.parent / "src/familiar_agent"


def _doc(name: str) -> str:
    """Return module docstring from tools/ or src/familiar_agent/ (whichever exists)."""
    for base in (_TOOLS, _SRC):
        p = base / name
        if p.exists():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            return ast.get_docstring(tree) or ""
    return ""


def test_load_manifest_returns_yaml_string():
    text = load_manifest()
    assert isinstance(text, str)
    assert len(text) > 0
    assert "capabilities" in text


def test_load_manifest_contains_known_capability():
    text = load_manifest()
    assert "autonomous_initiation" in text
    assert "memory" in text


def test_save_and_load_summary_roundtrip():
    save_summary("- I can recall memories.\n- I can speak autonomously.")
    result = load_summary()
    assert "recall memories" in result
    assert "autonomously" in result


def test_save_summary_overwrites_previous():
    save_summary("first summary")
    save_summary("second summary")
    result = load_summary()
    assert "second summary" in result
    assert "first summary" not in result


def test_load_summary_returns_empty_when_missing():
    # After truncation by conftest, agent_state is empty
    result = load_summary()
    assert result == ""


def test_camera_has_ptz():
    """camera.py docstring mentions look() or PTZ."""
    d = _doc("camera.py")
    assert any(k in d for k in ["look(", "PTZ", "pan-tilt"]), f"got: {d!r}"


def test_camera_has_see():
    """camera.py docstring mentions see()."""
    assert "see()" in _doc("camera.py"), f"got: {_doc('camera.py')!r}"


def test_tts_has_silent_mode():
    """tts.py docstring mentions silent/display-only mode."""
    d = _doc("tts.py").lower()
    assert any(k in d for k in ["silent", "display-only"]), f"got: {d!r}"


def test_mobility_has_walk():
    """mobility.py docstring mentions walk()."""
    assert "walk(" in _doc("mobility.py"), f"got: {_doc('mobility.py')!r}"


def test_person_has_declare_speaker():
    """person.py docstring lists declare_speaker."""
    assert "declare_speaker" in _doc("person.py"), f"got: {_doc('person.py')!r}"


def test_memory_has_remember_recall():
    """memory.py docstring mentions remember() and recall()."""
    d = _doc("memory.py")
    assert "remember(" in d and "recall(" in d, f"got: {d!r}"


def test_memory_worker_has_embedding():
    """memory_worker.py docstring mentions embedding or pgvector."""
    d = _doc("memory_worker.py").lower()
    assert any(k in d for k in ["embed", "pgvector"]), f"got: {d!r}"


# ── 一覧は `capabilities.yaml` だけ（環-y・2026-09-24） ──────────────────
#
# 一覧を DB（`agent_state.capabilities`）にも置ける作りだったが、**10 日たっても行は
# 一度も書かれず**、書く口（`regenerate_manifest`）も一度も走らなかった。実際に効いて
# いたのは `capabilities.yaml` だけで、空の器を経由して同じファイルを読んでいた。
#
# 一覧を書くのは**機能を作った人**である（`CLAUDE.md`：ファイルに置くのは既定値と人の
# 入力だけ）。機械に書き直させる案は本人が退けた——動いているアプリがリポジトリを
# 書き換えることになり、出来を誰も見ないため（2026-09-24）。


def test_the_capabilities_come_from_the_file():
    from familiar_agent.capability_state import load_capabilities, load_manifest

    assert load_capabilities() == load_manifest()


def test_the_db_layer_is_gone():
    """**旧名が残っていないこと**が撤去の証明である（数え上げでは代えられない）。"""
    import familiar_agent.capability_state as cs

    for name in ("store_capabilities", "capabilities_updated_at", "_CAPS_KEY"):
        assert not hasattr(cs, name), name


def test_the_file_is_never_written_at_runtime():
    import familiar_agent.capability_state as cs

    for name in (
        "save_manifest",
        "should_regenerate_manifest",
        "should_regenerate_on_startup",
        "should_refresh",
    ):
        assert not hasattr(cs, name), name
