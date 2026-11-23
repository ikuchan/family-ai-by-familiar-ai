"""定点一覧の組み立て（Config ＋ カメラのプリセット）。

定点は S3（在席マップ）・S4（norm）・S5（見回り）が共有するので、**一箇所で組んで配る**。
プリセットは、カメラの ONVIF が繋がったときに読む（人がカメラのアプリで足したものが、次の起動から定点になる）。

読めなかったときに空へ落ちるのは意図した degrade である。定点ゼロでも見ること自体はできる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.poses import Pose, build_pose_registry, parse_poses


def _camera(presets):
    cam = MagicMock()
    cam.presets = AsyncMock(return_value=presets)
    return cam


def test_the_registry_is_the_union_of_config_and_presets():
    got = asyncio.run(
        build_pose_registry(
            "窓側:0.0,-0.5;出入り口:-0.129,-0.5",
            _camera([Pose("台所", 0.8, -0.4)]),
            0.05,
        )
    )[0]
    assert {p.name for p in got} == {"窓側", "出入り口", "台所"}


def test_a_preset_at_a_configured_pose_does_not_double_it():
    """撮り直す前の実機のプリセットは pan=-0.1294・tilt=-0.2857 だった。

    距離を角度に揃える前は、tilt の 0.214 差がしきい値 0.05 を超えて別の定点になっていた。
    角度で測ると 7.5° で、しきい値の 8.5°（pan 換算）に収まるので**畳まれる**。同じ場所を
    高さ違いで2度見ることにはならない。
    """
    got = asyncio.run(
        build_pose_registry(
            "出入り口:-0.129,-0.5",
            _camera([Pose("出入り口", -0.1294, -0.2857)]),
            0.05,
        )
    )[0]
    assert len(got) == 1


def test_without_a_camera_only_the_configured_poses_remain():
    got, read = asyncio.run(build_pose_registry("窓側:0.0,-0.5", None, 0.05))
    assert [p.name for p in got] == ["窓側"] and read  # カメラが無ければ「読めた」


def test_a_camera_that_cannot_list_presets_still_yields_the_configured_poses():
    cam = MagicMock()
    cam.presets = AsyncMock(side_effect=RuntimeError("no ptz"))
    got, read = asyncio.run(build_pose_registry("窓側:0.0,-0.5", cam, 0.05))
    assert [p.name for p in got] == ["窓側"]
    assert not read  # 読めなかったことを返す（呼び手は、覚えた定点をこの読みで上書きしない）


def test_no_configuration_and_no_camera_is_empty_rather_than_an_error():
    assert asyncio.run(build_pose_registry("", None, 0.05)) == ([], True)


# ── カメラの ONVIF が繋がった事象で、定点を読んで配る（知-bw・2026-11-24）──────────────
#
# 起動のときにカメラがまだ繋がらないと、プリセットが読めず、定点 0 件のまま覚えて在席センサが黙って止まった（11/23 の
# 22:12・23:04・23:56 の起動）。そのあと ONVIF が繋がっても、誰も読み直さなかった。


def _agent_with(camera, config_poses=""):
    """定点まわりだけを動かす最小の器。`_poses` は `__init__` と同じく、設定の定点から始める。"""
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._camera = camera
    a.config = MagicMock()
    a.config.camera.poses = config_poses
    a.config.camera.pose_tolerance = 0.05
    a._poses = parse_poses(config_poses)
    return a


def _cam(presets_seq):
    cam = MagicMock()
    cam.presets = AsyncMock(side_effect=list(presets_seq))
    cam._ensure_connected = AsyncMock(return_value=False)
    return cam


def test_asking_for_poses_never_waits_for_the_camera():
    """定点を聞く道（反復、在席センサの 3 秒ごとの確かめ）は、覚えたものを受け取るだけで、カメラに聞かない。"""
    cam = _cam([[Pose("正面", 0.0, 0.0)]])
    a = _agent_with(cam)
    assert asyncio.run(a.poses()) == []
    cam.presets.assert_not_awaited()
    cam._ensure_connected.assert_not_awaited()


def test_poses_are_read_when_the_camera_connects():
    """起動のときは繋がらず 0 件。あとで ONVIF が繋がった事象で読み、`look` の行き先としても配る。"""
    cam = _cam([[Pose("正面", 0.0, 0.0), Pose("窓", 0.5, 0.0)]])
    a = _agent_with(cam)
    assert asyncio.run(a.poses()) == []
    asyncio.run(a._on_camera_connected())
    got = asyncio.run(a.poses())
    assert [p.name for p in got] == ["正面", "窓"]
    cam.set_poses.assert_called_once_with(got)


def test_late_poses_show_up_in_the_look_definition():
    """遅れて届いた定点が、道具の定義（`look` の行き先）に出る。定義は聞かれるたびに組むので、起動のときに控えた形は残らない。"""
    from familiar_agent.tools.camera import CameraTool

    cam = CameraTool.__new__(CameraTool)  # 取り込みの仕事を立てない
    cam._poses = []
    cam.presets = AsyncMock(return_value=[Pose("正面", 0.0, 0.0), Pose("窓", 0.5, 0.0)])
    a = _agent_with(cam)
    assert [d["name"] for d in cam.get_tool_definitions()] == ["see"]
    asyncio.run(a._on_camera_connected())
    look = next(d for d in cam.get_tool_definitions() if d["name"] == "look")
    assert look["input_schema"]["properties"]["pose"]["enum"] == ["正面", "窓"]


def test_presets_that_could_not_be_read_do_not_replace_known_poses():
    """**反証の側**：繋ぎ直したときにプリセットが読めなくても（None）、覚えた定点を 0 件で上書きしない。"""
    cam = _cam([[Pose("正面", 0.0, 0.0), Pose("窓", 0.5, 0.0)], None])
    a = _agent_with(cam)
    asyncio.run(a._on_camera_connected())
    asyncio.run(a._on_camera_connected())
    assert [p.name for p in asyncio.run(a.poses())] == ["正面", "窓"]


def test_a_camera_with_truly_no_presets_is_not_asked_again():
    """**反証の側**：読めて 0 件（本当にプリセットが無い）なら、そのあと何度定点を聞かれても、カメラに聞き直さない。"""
    cam = _cam([[]])
    a = _agent_with(cam)
    asyncio.run(a._on_camera_connected())
    for _ in range(3):
        assert asyncio.run(a.poses()) == []
    assert cam.presets.await_count == 1


def test_config_poses_do_not_hide_late_presets():
    """設定に定点があっても、ONVIF が繋がったらプリセットを足す。"""
    cam = _cam([[Pose("窓", 0.5, 0.0)]])
    a = _agent_with(cam, "正面:0.0,0.0")
    assert [p.name for p in asyncio.run(a.poses())] == ["正面"]
    asyncio.run(a._on_camera_connected())
    assert sorted(p.name for p in asyncio.run(a.poses())) == ["正面", "窓"]


# ── 起動のときに繋がらなかった ONVIF は、動体イベントの見張りが間をあけて取り直す ──────────────
#
# 見張りは口が None を返すと「カメラが無い構成」と見て黙って終わる。以前は「いま繋がらない」も None だったので、
# 11/23 の 3 回の起動では動体イベントも止まり、ONVIF を繋ぎに行く者が居なくなった。


def _onvif_camera(*, connected=False, fresh=True, connects=False):
    cam = MagicMock()
    cam.is_pan_tilt_available = True
    cam._cam_onvif = MagicMock(name="onvif") if connected else None
    cam.frame_is_fresh = MagicMock(return_value=fresh)

    async def _connect(**_kw):
        if connects:
            cam._cam_onvif = MagicMock(name="onvif")
        return connects

    cam._ensure_connected = AsyncMock(side_effect=_connect)
    return cam


def test_without_onvif_the_motion_watch_gets_none():
    """カメラが無い、ONVIF を持たない構成（USB など）は None。見張りは黙って終わり、聞き続けない。"""
    a = _agent_with(None)
    assert asyncio.run(a._connected_onvif()) is None
    usb = _onvif_camera()
    usb.is_pan_tilt_available = False
    a = _agent_with(usb)
    assert asyncio.run(a._connected_onvif()) is None
    usb._ensure_connected.assert_not_awaited()


def test_a_camera_that_is_not_up_yet_is_an_error_not_none():
    """映像が届いていない＝カメラごと落ちている。None で終わらせず例外にし（見張りが取り直す）、ONVIF は聞かない。"""
    cam = _onvif_camera(fresh=False)
    a = _agent_with(cam)
    with pytest.raises(ConnectionError):
        asyncio.run(a._connected_onvif())
    cam._ensure_connected.assert_not_awaited()


def test_frames_without_onvif_are_retried_quietly():
    """映像は届くのに ONVIF が繋がらない。例外にして取り直す。失敗の記録を下げるよう、取り直しだと伝える。"""
    cam = _onvif_camera(fresh=True, connects=False)
    a = _agent_with(cam)
    with pytest.raises(ConnectionError):
        asyncio.run(a._connected_onvif())
    cam._ensure_connected.assert_awaited_once_with(retrying=True)


def test_onvif_is_connected_once_frames_arrive():
    cam = _onvif_camera(fresh=True, connects=True)
    a = _agent_with(cam)
    assert asyncio.run(a._connected_onvif()) is cam._cam_onvif is not None


def test_a_connected_onvif_is_handed_over_as_before():
    """**反証の側**：もう繋がっていれば、映像を見ずにそのまま渡す（ふつうの起動は ONVIF が映像より先に繋がる）。"""
    cam = _onvif_camera(connected=True, fresh=False)
    a = _agent_with(cam)
    assert asyncio.run(a._connected_onvif()) is cam._cam_onvif
    cam._ensure_connected.assert_not_awaited()
