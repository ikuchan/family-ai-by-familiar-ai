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
    got = asyncio.run(build_pose_registry(
        "窓側:0.0,-0.5;出入り口:-0.129,-0.5",
        _camera([Pose("台所", 0.8, -0.4)]),
        0.05,
    ))
    assert {p.name for p in got} == {"窓側", "出入り口", "台所"}


def test_a_preset_at_a_configured_pose_does_not_double_it():
    """撮り直す前の実機のプリセットは pan=-0.1294・tilt=-0.2857 だった。

    距離を角度に揃える前は、tilt の 0.214 差がしきい値 0.05 を超えて別の定点になっていた。
    角度で測ると 7.5° で、しきい値の 8.5°（pan 換算）に収まるので**畳まれる**。同じ場所を
    高さ違いで2度見ることにはならない。
    """
    got = asyncio.run(build_pose_registry(
        "出入り口:-0.129,-0.5", _camera([Pose("出入り口", -0.1294, -0.2857)]), 0.05,
    ))
    assert len(got) == 1


def test_without_a_camera_only_the_configured_poses_remain():
    got = asyncio.run(build_pose_registry("窓側:0.0,-0.5", None, 0.05))
    assert [p.name for p in got] == ["窓側"]


def test_a_camera_that_cannot_list_presets_still_yields_the_configured_poses():
    cam = MagicMock()
    cam.presets = AsyncMock(side_effect=RuntimeError("no ptz"))
    got = asyncio.run(build_pose_registry("窓側:0.0,-0.5", cam, 0.05))
    assert [p.name for p in got] == ["窓側"]


def test_no_configuration_and_no_camera_is_empty_rather_than_an_error():
    assert asyncio.run(build_pose_registry("", None, 0.05)) == []
