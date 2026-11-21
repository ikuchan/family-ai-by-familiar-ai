"""環-bc の根元：在席の見回りの画像は、いつも同じ 1 つのファイルに書き直す。

見回り（3 秒おき）が撮るたびに時刻つきの名前で書き、1 日に約 1 万 3 千枚（約 1.5 GB）が増えていた。見回りの画像を
使うのは同じ 1 回の確かめの中だけ（人の検出と見えの埋め込み）なので、残さなくてよい。見る道具（`look`・`see`）は、
いままでどおり時刻つきの名前で残し、主LLM に見せる「直近に撮った画像」（`last_capture_path`）になる。
"""

from __future__ import annotations

import asyncio

import numpy as np

from tests.test_camera import _make_camera_tool


def test_patrol_frames_overwrite_one_file_and_do_not_become_the_last_capture(
    tmp_path, monkeypatch
) -> None:
    import familiar_agent.tools.camera as camera_module

    monkeypatch.setattr(camera_module, "CAPTURE_DIR", tmp_path)
    cam = _make_camera_tool()
    cam.last_capture_path = None
    cam._last_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    _, first = asyncio.run(cam.capture(patrol=True))
    _, second = asyncio.run(cam.capture(patrol=True))
    assert first == second and first is not None
    assert [p.name for p in tmp_path.iterdir()] == ["patrol.jpg"]
    assert cam.last_capture_path is None


def test_a_looked_frame_still_gets_its_own_name(tmp_path, monkeypatch) -> None:
    import familiar_agent.tools.camera as camera_module

    monkeypatch.setattr(camera_module, "CAPTURE_DIR", tmp_path)
    cam = _make_camera_tool()
    cam._last_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    _, path = asyncio.run(cam.capture())
    assert path is not None and path.split("/")[-1].startswith("capture_")
    assert cam.last_capture_path == path


def test_the_occupancy_sensor_asks_for_a_patrol_frame() -> None:
    from tests.test_occupancy_sensor import _sensor

    s, camera, _ = _sensor(people=1)
    asyncio.run(s.check_once())
    camera.capture.assert_awaited_once_with(patrol=True)
