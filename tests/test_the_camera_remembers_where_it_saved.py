"""撮った画像の在りかをカメラが覚えている（`イベント駆動ループ` v0.43）。

見た印がこれを持ち、その求めのあいだ主LLM が画像そのものを見る。`call()` の返りは
変えない（呼び手3箇所と既存テストを壊さない）。
"""

from __future__ import annotations

import pytest

from tests.test_camera import _make_camera_tool, _make_fake_frame


@pytest.mark.asyncio
async def test_capture_remembers_the_saved_path(monkeypatch, tmp_path) -> None:
    import familiar_agent.tools.camera as mod

    monkeypatch.setattr(mod, "CAPTURE_DIR", tmp_path)
    cam = _make_camera_tool()
    cam.last_capture_path = None
    cam._last_frame = _make_fake_frame()

    _b64, path = await cam.capture()

    assert path and cam.last_capture_path == path
    assert (tmp_path / path.split("/")[-1]).exists()


@pytest.mark.asyncio
async def test_a_failed_capture_leaves_the_last_path_as_it_was() -> None:
    cam = _make_camera_tool()
    cam.last_capture_path = None
    cam._last_frame = None

    await cam.capture()

    assert cam.last_capture_path is None
