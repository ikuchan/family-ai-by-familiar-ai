"""首を振る動作を、定点名で呼ぶ形に替える。

`ユースケース③` は見回りの1ステップを「**絶対 pan/tilt で1定点へ移動** → 観察」と定める。
相対の首振り（左へ30度）では定点から外れ、振動中ゲートに落ちて**在席も見えの「普通」も
更新されなくなる**（実機で、カメラ側の追尾が向きを変えたときに実際に起きた）。相対の
首振りは定点の仕組みと共存できないので、置き換える。

定点名は道具の定義に `enum` として並べる。存在しない場所を選べなくなり、どこを見に行けるかが
そのまま伝わる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.poses import Pose
from familiar_agent.tools.camera import CameraTool

_POSES = [Pose("窓側", 0.0, -0.5), Pose("出入り口", -0.129, -0.5), Pose("襖側", -0.667, -0.29)]


def _tool(poses=_POSES):
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    cam.move_to = AsyncMock(return_value="Turned to pan=+0.0000 tilt=-0.5000.")
    cam.capture = AsyncMock(return_value=("B64", "/tmp/f.jpg"))
    cam.set_poses(poses)
    return cam


def _look_def(cam):
    return next(d for d in cam.get_tool_definitions() if d["name"] == "look")


# --- 道具の定義 -----------------------------------------------------------


def test_the_places_it_can_look_at_are_listed():
    schema = _look_def(_tool())["input_schema"]
    assert schema["properties"]["pose"]["enum"] == ["窓側", "出入り口", "襖側"]


def test_the_place_is_required():
    assert _look_def(_tool())["input_schema"]["required"] == ["pose"]


def test_the_old_relative_arguments_are_gone():
    # 相対の首振りは撤去した（定点から外れて振動中ゲートに落ちるため）。
    props = _look_def(_tool())["input_schema"]["properties"]
    assert "direction" not in props and "degrees" not in props


def test_without_poses_there_is_nothing_to_look_at():
    # 定点が無ければ首を振る先も無い。道具ごと出さない（選べない動作を見せない）。
    names = {d["name"] for d in _tool(poses=[]).get_tool_definitions()}
    assert "look" not in names
    assert "see" in names          # 見ること自体はできる


# --- 呼び出し -------------------------------------------------------------


def test_looking_at_a_place_moves_there_absolutely():
    cam = _tool()
    asyncio.run(cam.call("look", {"pose": "襖側"}))
    cam.move_to.assert_awaited_once_with(-0.667, -0.29)


def test_the_reply_names_the_place_it_turned_to():
    cam = _tool()
    text, image = asyncio.run(cam.call("look", {"pose": "窓側"}))
    assert "窓側" in text
    assert image is None           # 首を振っただけで画像は無い


def test_an_unknown_place_is_refused_rather_than_moving_somewhere_odd():
    cam = _tool()
    text, _ = asyncio.run(cam.call("look", {"pose": "台所"}))
    cam.move_to.assert_not_awaited()
    assert "台所" in text


# --- 求めの見出し ---------------------------------------------------------


def test_the_heading_names_the_place():
    from familiar_agent.loop.event_loop import _query_label

    assert _query_label("look", {"pose": "窓側"}) == "窓側を見に行く"


def test_two_places_get_different_headings():
    from familiar_agent.loop.event_loop import _query_label

    assert _query_label("look", {"pose": "窓側"}) != _query_label("look", {"pose": "襖側"})
