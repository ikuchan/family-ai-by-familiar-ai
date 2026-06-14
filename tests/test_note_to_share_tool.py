"""Tests for note_to_share tool in MemoryTool (Issue D).

観察記憶に紐づいた「話したいこと」を pending_speech へ登録するツール。
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from familiar_agent.tools.memory import MemoryTool
from familiar_agent.tools.pending_speech_store import PendingSpeechStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store():
    store = MagicMock(spec=PendingSpeechStore)
    store.add.return_value = str(uuid.uuid4())
    return store


@pytest.fixture()
def mock_manager():
    mgr = MagicMock()
    mgr.find_person_id_by_name.return_value = "person-123"
    mgr.current_speaker_id = None
    mgr.get_present_ids.return_value = []
    return mgr


@pytest.fixture()
def tool(mock_manager, mock_store):
    t = MemoryTool.__new__(MemoryTool)
    t._manager = mock_manager
    t._pending_store = mock_store
    return t


# ---------------------------------------------------------------------------
# Tests: note_to_share tool definition
# ---------------------------------------------------------------------------


def test_note_to_share_in_tool_definitions(tool):
    defs = tool.get_tool_definitions()
    names = [d["name"] for d in defs]
    assert "note_to_share" in names


def test_note_to_share_definition_has_observation_id_required(tool):
    defs = {d["name"]: d for d in tool.get_tool_definitions()}
    schema = defs["note_to_share"]["input_schema"]
    assert "observation_id" in schema["required"]


def test_note_to_share_definition_has_optional_target(tool):
    defs = {d["name"]: d for d in tool.get_tool_definitions()}
    schema = defs["note_to_share"]["input_schema"]
    assert "target" in schema["properties"]
    assert "target" not in schema.get("required", [])


# ---------------------------------------------------------------------------
# Tests: _note_to_share logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_note_to_share_resolves_target_name(tool, mock_manager, mock_store):
    """target 名前 → person_id 解決して store.add に渡す。"""
    mock_store.add.return_value = "new-pid"
    result, _ = await tool._note_to_share({"observation_id": "obs-1", "target": "ユーザー"})
    mock_manager.find_person_id_by_name.assert_called_once_with("ユーザー")
    mock_store.add.assert_called_once_with("obs-1", "person-123")
    assert "登録" in result


@pytest.mark.asyncio
async def test_note_to_share_unknown_name_falls_back_null(tool, mock_manager, mock_store):
    """未登録名 → NULL(誰でも)で登録（拒否しない）。"""
    mock_manager.find_person_id_by_name.return_value = None
    mock_store.add.return_value = "new-pid"
    result, _ = await tool._note_to_share({"observation_id": "obs-1", "target": "未登録の人"})
    mock_store.add.assert_called_once_with("obs-1", None)
    assert "登録" in result


@pytest.mark.asyncio
async def test_note_to_share_no_target_uses_null(tool, mock_manager, mock_store):
    """target 省略 → NULL(誰でも)で登録。"""
    mock_store.add.return_value = "new-pid"
    result, _ = await tool._note_to_share({"observation_id": "obs-1"})
    mock_store.add.assert_called_once_with("obs-1", None)
    assert "登録" in result


@pytest.mark.asyncio
async def test_note_to_share_rejects_unknown_observation(tool, mock_store):
    """存在しない observation_id → 拒否メッセージを返す。"""
    mock_store.add.return_value = None
    result, _ = await tool._note_to_share({"observation_id": "nonexistent"})
    assert "先に" in result or "覚えていない" in result


# ---------------------------------------------------------------------------
# Tests: dispatched via call()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_note_to_share_dispatched_via_call(tool, mock_store):
    mock_store.add.return_value = "pid-1"
    result, _ = await tool.call("note_to_share", {"observation_id": "obs-1"})
    assert isinstance(result, str)
