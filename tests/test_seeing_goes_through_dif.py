"""知覚（`see`・`look`）は DIF の口 `perceive` を通る（環-ac (1)・2026-11-23）。挙動は変えない。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.io.dif import DIF


def test_perceive_hands_the_call_to_the_camera_as_it_is():
    cam = MagicMock()
    cam.call = AsyncMock(return_value=("撮った", "b64"))
    dif = DIF(camera=lambda: cam)
    assert asyncio.run(dif.perceive("look", {"pose": "窓"})) == ("撮った", "b64")
    cam.call.assert_awaited_once_with("look", {"pose": "窓"})


def test_without_a_camera_it_fails_the_way_it_did():
    """カメラが無いときは、前と同じく `None.call` の AttributeError（呼び手 `_run_camera` が畳む）。"""
    dif = DIF(camera=lambda: None)
    with pytest.raises(AttributeError):
        asyncio.run(dif.perceive("see", {}))


def test_the_loop_reaches_the_camera_only_through_dif():
    """**反証側**：ループが `agent._camera.call` を直に呼んでいれば、DIF を通ったことにならない。"""
    import pathlib

    loop = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent" / "loop"
    files = sorted(loop.glob("*.py"))
    assert len(files) >= 30  # 拾えていなければ空振りで通る
    direct = [f.name for f in files if "_camera.call(" in f.read_text(encoding="utf-8")]
    assert direct == []
    assert "self._dif.perceive(" in (loop / "event_loop.py").read_text(encoding="utf-8")


def test_the_camera_read_late_is_the_one_swapped_in():
    """試験や再接続でカメラを後から差し替えても、口は差し替えた先を呼ぶ（作った時点のものを掴まない）。"""
    holder = MagicMock(_camera=None)
    dif = DIF(camera=lambda: holder._camera)
    cam = MagicMock()
    cam.call = AsyncMock(return_value=("見た", None))
    holder._camera = cam
    assert asyncio.run(dif.perceive("see", {})) == ("見た", None)
