"""起動したら自律が回り始める（人の発話を待たない）。

`感情ループ全体像` は「起動源は Drive（欲求）の時間蓄積と発火」と定める。ところが実装では
I（情報処理機構）も T（自律機構）も在席センサも動体イベントも、**すべて `run()` の中**、
それも `user_input` があるときにしか立たなかった。起動しても、人が話しかけるまで何ひとつ
回っていない。

これは3つの症状の同じ根である。

- `/speaker` を最初に打つと入室イベントが立たない（T がまだ起動していない）
- 在席が「連続」にならない（`知覚在席` §3-2 は G（T 側・連続）と定める）
- 保留していた発話を配る起点（在席がゼロから立ち上がる瞬間）が永久に来ない

`start_autonomy()` を新設し、GUI と CUI の両方の入口から呼ぶ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent


def _agent(*, sensor=True):
    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a.config = MagicMock()
    a._info_processing = None
    a._tonic = None
    a._occupancy_sensor = MagicMock(start=AsyncMock()) if sensor else None
    a._motion_events = MagicMock(start=AsyncMock()) if sensor else None
    # ループは生成のときに機器を DIF へ渡す。`__init__` を通さない土台なので、
    # 実機体が必ず持つものはここで置く（無い機体の `_tts` は None）。
    a._tts = None
    a._mcp = None
    a._deferred_search = MagicMock()
    a._deferred_fetch = MagicMock()
    return a


def test_the_autonomous_side_starts_without_anyone_speaking():
    a = _agent()
    asyncio.run(a.start_autonomy())
    assert a._tonic is not None
    assert a._info_processing is not None


def test_the_occupancy_sensor_starts_too():
    a = _agent()
    asyncio.run(a.start_autonomy())
    a._occupancy_sensor.start.assert_awaited()
    a._motion_events.start.assert_awaited()


def test_the_camera_is_connected_before_the_sensors_start():
    """起動のときに ONVIF を 1 回繋ぐ（繋がった事象で定点を読む・知-bw）。`look` の行き先が最初の反復に間に合うよう、先に待つ。"""
    a = _agent()
    order = []
    a._camera = MagicMock(_ensure_connected=AsyncMock(side_effect=lambda: order.append("camera")))
    a._occupancy_sensor.start = AsyncMock(side_effect=lambda: order.append("sensor"))
    asyncio.run(a.start_autonomy())
    assert order == ["camera", "sensor"]


def test_poses_are_read_through_the_start_when_the_camera_answers():
    """起動で口を渡してから繋ぐので、繋がった事象がエージェントに届き、最初の反復の前に定点が読めている（知-bw）。"""
    from familiar_agent.poses import Pose

    a = _agent()
    a._poses = []
    a.config.camera.poses = ""
    a.config.camera.pose_tolerance = 0.05
    cam = MagicMock()
    cam.presets = AsyncMock(return_value=[Pose("正面", 0.0, 0.0), Pose("窓", 0.5, 0.0)])

    async def connect():  # 本物のカメラと同じく、繋がったら渡された口を待つ
        await cam.on_connected()
        return True

    cam._ensure_connected = AsyncMock(side_effect=connect)
    a._camera = cam
    asyncio.run(a.start_autonomy())
    assert [p.name for p in asyncio.run(a.poses())] == ["正面", "窓"]
    cam.set_poses.assert_called_once()


def test_a_real_camera_tool_hands_its_presets_over_at_the_start(monkeypatch):
    """継ぎ目ごと確かめる：本物の `CameraTool`（ONVIF だけ偽物）を起動に通すと、接続、知らせ、プリセットの読み、定点、`look` の定義まで繋がる。"""
    import familiar_agent.tools.camera as camera_mod
    from familiar_agent.tools.camera import CameraTool

    def preset(name, pan, tilt):
        q = MagicMock()
        q.Name = name
        q.PTZPosition.PanTilt.x, q.PTZPosition.PanTilt.y = pan, tilt
        return q

    ptz = MagicMock()
    ptz.GetPresets = AsyncMock(return_value=[preset("正面", 0.0, 0.0), preset("窓", 0.5, 0.0)])

    class _FakeOnvif:
        def __init__(self, *_a, **_k) -> None:
            pass

        async def update_xaddrs(self) -> None:
            pass

        async def create_media_service(self):
            media = MagicMock()
            media.GetProfiles = AsyncMock(return_value=[MagicMock(token="profile_1")])
            return media

        async def create_ptz_service(self):
            return ptz

    monkeypatch.setattr(camera_mod, "ONVIFCamera", _FakeOnvif)
    cam = CameraTool.__new__(CameraTool)  # 取り込みの仕事を立てない
    cam.host = cam.ptz_host = "192.0.2.1"
    cam.username = cam.ptz_username = "u"
    cam.password = cam.ptz_password = "p"
    cam.port = cam.ptz_port = 2020
    cam._cam_onvif = cam._ptz = cam._profile_token = None
    cam._poses = []

    a = _agent()
    a._poses = []
    a.config.camera.poses = ""
    a.config.camera.pose_tolerance = 0.05
    a._camera = cam
    asyncio.run(a.start_autonomy())
    assert [p.name for p in asyncio.run(a.poses())] == ["正面", "窓"]
    look = next(d for d in cam.get_tool_definitions() if d["name"] == "look")
    assert look["input_schema"]["properties"]["pose"]["enum"] == ["正面", "窓"]
    assert ptz.GetPresets.await_count == 1  # ふつうの起動で読むのは 1 回


def test_a_camera_that_is_not_up_does_not_stop_the_start():
    """**反証の側**：起動のときに繋がらなくても（例外でも）、センサと見張りは立つ。あとは見張りが取り直す。"""
    a = _agent()
    a._camera = MagicMock(_ensure_connected=AsyncMock(side_effect=OSError("unreachable")))
    asyncio.run(a.start_autonomy())
    a._occupancy_sensor.start.assert_awaited()
    a._motion_events.start.assert_awaited()


def test_starting_twice_does_not_make_a_second_tonic():
    a = _agent()
    asyncio.run(a.start_autonomy())
    first = a._tonic
    asyncio.run(a.start_autonomy())
    assert a._tonic is first


def test_a_configuration_without_a_camera_still_starts():
    a = _agent(sensor=False)
    asyncio.run(a.start_autonomy())  # 例外を出さないこと
    assert a._tonic is not None
