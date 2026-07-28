"""カメラの絶対 pan/tilt（定点へ向く・いまどこを向いているか）。

見回りは「絶対 pan/tilt で1定点へ移動 → 観察」を1反復1ステップで行う（`ユースケース③`）。
既存の `move()` は相対（左へ30度）で、定点へ向けない。実機の ONVIF は
`AbsolutePanTiltPositionSpace` を pan・tilt とも $[-1,1]$ で持つことを確認済み。

在席と norm を定点ごとに持つには、**いまどこを向いているか**も要る。動いている最中は
どの定点でもないので、更新を止める（振動中ゲート）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.poses import Pose
from familiar_agent.tools.camera import CameraTool


def _tool(*, status_xy=(-0.129, -0.50), presets=None):
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    ptz = MagicMock()
    ptz.AbsoluteMove = AsyncMock(return_value=None)
    st = MagicMock()
    st.Position.PanTilt.x, st.Position.PanTilt.y = status_xy
    ptz.GetStatus = AsyncMock(return_value=st)
    ptz.GetPresets = AsyncMock(return_value=presets if presets is not None else [])
    cam._ptz = ptz
    cam._profile_token = "profile_1"
    cam._cam_onvif = MagicMock()
    return cam, ptz


def _preset(token, name, pan, tilt):
    q = MagicMock()
    q.token = token
    q.Name = name
    q.PTZPosition.PanTilt.x, q.PTZPosition.PanTilt.y = pan, tilt
    return q


# --- 絶対移動 -------------------------------------------------------------


def test_moving_to_a_pose_sends_an_absolute_position():
    cam, ptz = _tool()
    asyncio.run(cam.move_to(-0.667, -0.29))
    sent = ptz.AbsoluteMove.call_args[0][0]
    assert sent["Position"]["PanTilt"] == {"x": -0.667, "y": -0.29}
    assert sent["ProfileToken"] == "profile_1"


def test_moving_outside_the_range_is_clamped():
    # 範囲外を送るとカメラが拒否する。手前で丸めて、動かないより動かす。
    cam, ptz = _tool()
    asyncio.run(cam.move_to(2.0, -3.0))
    assert ptz.AbsoluteMove.call_args[0][0]["Position"]["PanTilt"] == {"x": 1.0, "y": -1.0}


def test_a_failed_move_reports_instead_of_raising():
    cam, ptz = _tool()
    ptz.AbsoluteMove = AsyncMock(side_effect=RuntimeError("ptz offline"))
    out = asyncio.run(cam.move_to(0.0, -0.5))
    assert "ptz offline" in out


# --- 現在の向き -----------------------------------------------------------


def test_the_current_position_is_readable():
    cam, _ = _tool(status_xy=(-0.1997, -0.6572))
    assert asyncio.run(cam.position()) == (-0.1997, -0.6572)


def test_an_unreadable_position_is_none_rather_than_a_guess():
    # 分からないときに 0,0 を返すと、向いていない定点を向いていることにしてしまう。
    cam, ptz = _tool()
    ptz.GetStatus = AsyncMock(side_effect=RuntimeError("no status"))
    assert asyncio.run(cam.position()) is None


# --- プリセット -----------------------------------------------------------


def test_presets_are_read_as_poses():
    cam, _ = _tool(presets=[_preset("1", "出入り口", -0.1294, -0.2857)])
    got = asyncio.run(cam.presets())
    assert got == [Pose("出入り口", -0.1294, -0.2857)]


def test_a_preset_without_a_position_is_skipped():
    q = MagicMock()
    q.token, q.Name, q.PTZPosition = "2", "位置なし", None
    cam, _ = _tool(presets=[q])
    assert asyncio.run(cam.presets()) == []


def test_presets_degrade_to_empty_when_the_camera_refuses():
    cam, ptz = _tool()
    ptz.GetPresets = AsyncMock(side_effect=RuntimeError("no presets"))
    assert asyncio.run(cam.presets()) == []
