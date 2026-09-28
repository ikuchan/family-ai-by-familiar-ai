"""Tests for extract_entities — entity extraction from a scene description.

（場面の実体を DB に追う `SceneTracker` の試験は、仕組みごと環-ab で外した。）
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.scene import extract_entities


def _backend_with_response(json_response: str) -> MagicMock:
    backend = MagicMock()
    backend.complete = AsyncMock(return_value=json_response)
    return backend


# ---------------------------------------------------------------------------
# Tests: extract_entities()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_entities_calls_backend():
    """extract_entities() must call backend.complete with the description."""
    backend = _backend_with_response('{"entities": []}')

    await extract_entities("A room with a chair.", backend)

    backend.complete.assert_awaited_once()
    prompt_arg = backend.complete.call_args[0][0]
    assert "A room with a chair." in prompt_arg


@pytest.mark.asyncio
async def test_extract_entities_returns_list():
    """extract_entities() returns a list of entity dicts."""
    payload = json.dumps(
        {
            "entities": [
                {"label": "chair", "category": "object", "confidence": 0.9},
                {"label": "person", "category": "person", "confidence": 0.85},
            ]
        }
    )
    backend = _backend_with_response(payload)

    result = await extract_entities("A room with a chair and person.", backend)

    assert len(result) == 2
    labels = {e["label"] for e in result}
    assert labels == {"chair", "person"}


@pytest.mark.asyncio
async def test_extract_entities_handles_malformed_json():
    """extract_entities() returns empty list when backend returns non-JSON."""
    backend = _backend_with_response("Sorry, I can't parse that.")

    result = await extract_entities("Some description.", backend)

    assert result == []


@pytest.mark.asyncio
async def test_extract_entities_handles_missing_entities_key():
    """extract_entities() returns empty list when JSON has no 'entities' key."""
    backend = _backend_with_response('{"result": "ok"}')

    result = await extract_entities("Some description.", backend)

    assert result == []


# ---------------------------------------------------------------------------
# Tests: SceneTracker.update() — entity storage
# ---------------------------------------------------------------------------
