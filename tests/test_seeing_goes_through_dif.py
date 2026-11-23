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


def test_the_loop_reads_the_camera_and_the_presence_sensor_through_dif():
    """環-ac (2)-a（2026-11-23）：I はカメラと在席センサを DIF から引く（設計図の案ア＝T は直接読み・I は DIF 経由）。

    見るのは `event_loop.py`（ループの持ち主）だけ。T（`tonic.py`）は案アのとおり直接読むので見ない。agent を受け取る
    モジュール関数 `actions._camera_tool_def`・`generator._present_ctx` は、ループの状態も DIF も持たないので外した。
    """
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "familiar_agent"
        / "loop"
        / "event_loop.py"
    ).read_text(encoding="utf-8")
    # 許すのは、DIF を作るときに引き口を渡す 1 か所ずつだけ（`camera=lambda: …`・`sensor=lambda: …`）。
    assert src.count('getattr(agent, "_occupancy_sensor"') == 1
    assert src.count('getattr(self._agent, "_occupancy_sensor"') == 0
    assert "agent._camera" not in src and src.count('getattr(agent, "_camera"') == 1
    assert "self._dif.sensor()" in src and "self._dif.camera()" in src


def test_dif_reads_the_sensor_late():
    """センサを後から差し替えても（試験・起動の順）、口は差し替えた先を返す。"""
    holder = MagicMock(_occupancy_sensor=None)
    dif = DIF(sensor=lambda: holder._occupancy_sensor)
    assert dif.sensor() is None
    holder._occupancy_sensor = sensor = object()
    assert dif.sensor() is sensor
