"""Tests for capability_state — manifest loader and AI self-understanding storage."""

from __future__ import annotations

import ast
from pathlib import Path

from familiar_agent.capability_state import (
    build_generation_prompt,
    collect_manifest_context,
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


def test_collect_manifest_context_returns_string():
    result = collect_manifest_context()
    assert isinstance(result, str)
    assert len(result) > 0


def test_collect_manifest_context_contains_sections():
    result = collect_manifest_context()
    assert "Built-in tools" in result
    assert "Key modules" in result
    assert ".env" in result
    assert "MCP servers" in result


def test_collect_manifest_context_contains_known_tool():
    result = collect_manifest_context()
    # memory.py and tts.py are always present
    assert "memory.py" in result or "tools/memory" in result


def test_collect_manifest_context_redacts_secrets():
    result = collect_manifest_context()
    assert "API_KEY=<redacted>" in result or "API_KEY" not in result or "<redacted>" in result


# ---------------------------------------------------------------------------
# build_generation_prompt
# ---------------------------------------------------------------------------


def test_build_generation_prompt_contains_context():
    prompt = build_generation_prompt("## test context\nsome info", "")
    assert "test context" in prompt
    assert "some info" in prompt


def test_build_generation_prompt_includes_existing_yaml():
    prompt = build_generation_prompt("ctx", "capabilities:\n  - id: memory")
    assert "memory" in prompt


def test_build_generation_prompt_omits_existing_when_empty():
    prompt = build_generation_prompt("ctx", "")
    assert "Existing capabilities.yaml" not in prompt


def test_build_generation_prompt_instructs_yaml_output():
    prompt = build_generation_prompt("ctx", "")
    assert "capabilities:" in prompt
    assert "YAML" in prompt


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


# ── 層 4 の器：一覧は DB（既定は capabilities.yaml・実行時に書き換えない）（記-a-と・2026-09-14） ──


def test_the_capabilities_come_from_the_file_until_the_db_has_a_version():
    from familiar_agent.capability_state import load_capabilities, load_manifest, store_capabilities

    assert load_capabilities() == load_manifest()  # DB に無ければ既定
    store_capabilities("capabilities:\n  - name: x\n")
    assert load_capabilities() == "capabilities:\n  - name: x\n"
    assert load_manifest() != load_capabilities()  # file は書き換わらない


def test_the_file_is_never_written_at_runtime():
    import familiar_agent.capability_state as cs

    for name in (
        "save_manifest",
        "should_regenerate_manifest",
        "should_regenerate_on_startup",
        "should_refresh",
    ):
        assert not hasattr(cs, name), name
