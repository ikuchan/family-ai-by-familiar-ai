"""`look` に相対の向き（右／左／上／下）を足す（知-m ③・2026-09-16）。

定点は pan/tilt を持つので、関係を人が書かなくても「いまの向きから右で最も近い定点」は機械で決まる。
右＝pan が大きい側〔仮・実機で確かめる。逆なら `RIGHT_IS_POSITIVE_PAN` を反転〕。名前の無い向きへは行かない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from familiar_agent.poses import Pose, pose_toward
from familiar_agent.tools.camera import CameraTool

POSES = [
    Pose("出入口", -0.548, 0.143),
    Pose("押入", -0.254, -0.143),
    Pose("正面", -0.078, -0.143),
    Pose("窓", 0.194, -0.143),
    Pose("テレビ", 0.491, 0.485),
]


def test_the_nearest_pose_in_each_direction_is_chosen():
    assert pose_toward(POSES, (-0.078, -0.143), "右").name == "窓"
    assert pose_toward(POSES, (-0.078, -0.143), "左").name == "押入"
    assert (
        pose_toward(POSES, (-0.078, -0.143), "上").name == "出入口"
    )  # tilt が大きい側で、向きとして最も近い
    assert pose_toward(POSES, (0.491, 0.485), "右") is None  # 右端
    assert pose_toward(POSES, (0.491, 0.485), "下").name == "窓"
    assert pose_toward(POSES, (-0.548, 0.143), "左") is None


def test_a_tiny_step_does_not_count_as_that_direction():
    # いまの向きとほぼ同じ pan の定点は「右」ではない（誤差の吸収・tolerance）。
    assert pose_toward(POSES, (-0.08, -0.143), "右", tolerance=0.02).name == "窓"


def _cam(position):
    cam = CameraTool.__new__(CameraTool)
    cam._poses = list(POSES)
    cam.position = AsyncMock(return_value=position)
    cam.move_to = AsyncMock(return_value="ok")
    return cam


def test_look_accepts_a_direction_and_reports_where_it_went():
    cam = _cam((-0.078, -0.143))
    text, _ = asyncio.run(cam.call("look", {"direction": "右"}))
    assert "窓" in text
    cam.move_to.assert_awaited_once_with(0.194, -0.143)


def test_look_says_when_nothing_is_further_that_way():
    cam = _cam((0.491, 0.485))
    text, _ = asyncio.run(cam.call("look", {"direction": "右"}))
    assert "右" in text and "知っている" in text and "テレビ" in text
    cam.move_to.assert_not_awaited()


def test_look_says_when_the_current_direction_is_unknown():
    cam = _cam(None)
    text, _ = asyncio.run(cam.call("look", {"direction": "左"}))
    assert "分からない" in text
    cam.move_to.assert_not_awaited()


def test_the_tool_definition_offers_both_pose_and_direction():
    cam = _cam((0, 0))
    look = next(d for d in cam.get_tool_definitions() if d["name"] == "look")
    props = look["input_schema"]["properties"]
    assert set(props["pose"]["enum"]) == {p.name for p in POSES}
    assert props["direction"]["enum"] == ["右", "左", "上", "下"]
    assert look["input_schema"].get("required") in (None, [])
