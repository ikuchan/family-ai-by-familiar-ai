"""定点一覧の組み立て（Config ＋ カメラのプリセット）。

定点は S3（在席マップ）・S4（norm）・S5（見回り）が共有するので、**一箇所で組んで配る**。
プリセットは起動のたびに読む（人がカメラのアプリで足したものが、次の起動から定点になる）。

読めなかったときに空へ落ちるのは意図した degrade である。定点ゼロでも見ること自体はできる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.poses import Pose, build_pose_registry


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
    assert [p.name for p in got] == [
        "窓側"
    ] and not read  # 読めなかったことを返す（読み直せるように）


def test_no_configuration_and_no_camera_is_empty_rather_than_an_error():
    assert asyncio.run(build_pose_registry("", None, 0.05)) == ([], True)


# ── 読めなかったプリセットは、映像が戻ってから読み直す（知-bw・2026-11-24）──────────────


def _cam(presets_seq, *, fresh=True):
    cam = MagicMock()
    cam.presets = AsyncMock(side_effect=list(presets_seq))
    cam.frame_is_fresh = MagicMock(return_value=fresh)
    return cam


def _agent_with(camera, now):
    """`EmbodiedAgent.poses` だけを動かす最小の器（定点の設定は空・プリセットはカメラから）。"""
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._poses = None
    a._camera = camera
    a.config = MagicMock()
    a.config.camera.poses = ""
    a.config.camera.pose_tolerance = 0.05
    a.config.camera.pose_retry_sec = 300.0
    a._clock = lambda: now[0]
    return a


def test_unread_presets_are_read_again_once_frames_come_back(caplog):
    """起動のときにカメラがまだ繋がらずプリセットが読めないと、0 件のまま覚えて在席センサが黙って止まった（11/23 の 3 回の起動）。

    読めなかったときは、映像が届いていて、前の試しから `pose_retry_sec` たったら 1 回読み直す。
    """
    now = [0.0]
    cam = _cam([None, [Pose("正面", 0.0, 0.0), Pose("窓", 0.5, 0.0)]])
    a = _agent_with(cam, now)
    assert asyncio.run(a.poses()) == []
    now[0] = 100.0
    assert asyncio.run(a.poses()) == []  # まだ 5 分たっていないので聞かない
    assert cam.presets.await_count == 1
    now[0] = 301.0
    assert [p.name for p in asyncio.run(a.poses())] == ["正面", "窓"]
    assert cam.presets.await_count == 2


def test_no_retry_while_the_camera_sends_no_frames():
    """カメラが落ちているあいだ（映像が来ない）は聞きに行かない（接続を待ってループを止めない・ログを埋めない）。"""
    now = [0.0]
    cam = _cam([None, [Pose("正面", 0.0, 0.0)]], fresh=False)
    a = _agent_with(cam, now)
    asyncio.run(a.poses())
    now[0] = 1000.0
    assert asyncio.run(a.poses()) == []
    assert cam.presets.await_count == 1


def test_a_camera_with_truly_no_presets_is_not_asked_again():
    """**反証の側**：読めて 0 件（本当にプリセットが無い）なら覚え、聞き直さない。"""
    now = [0.0]
    cam = _cam([[]])
    a = _agent_with(cam, now)
    asyncio.run(a.poses())
    now[0] = 10_000.0
    asyncio.run(a.poses())
    assert cam.presets.await_count == 1


def test_config_poses_do_not_hide_unread_presets():
    """設定に定点があっても、プリセットが読めなかったら、映像が戻ったあとに読み直す。"""
    now = [0.0]
    cam = _cam([None, [Pose("窓", 0.5, 0.0)]])
    a = _agent_with(cam, now)
    a.config.camera.poses = "正面:0.0,0.0"
    assert [p.name for p in asyncio.run(a.poses())] == ["正面"]
    now[0] = 301.0
    assert sorted(p.name for p in asyncio.run(a.poses())) == ["正面", "窓"]
