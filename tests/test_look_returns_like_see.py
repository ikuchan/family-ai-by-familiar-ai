"""`look` の帰りは `see` と同じ扱い（2026-09-16 実機 15:11）。

「右向いて」に調停が `look` を選び、首は 0.5 秒で回った。ところが帰りの調停が `look` を
3 回選び直し（同語なので機械が投げない）、上限で主LLM が「はーい、右向いてみるね！」と
言うまで 18 秒かかった。`see` には帰りの扱い（誰が出したかを控える・帰りは出した側が判断
する・写真と但し書きを渡す・見た印を残す）があり、`look` には無かった。**首を向けただけで、
向いた先を誰も見ていなかった**ので、調停は「右を見る」がまだ済んでいないと読んだ。

`look` は首を向けたあとにその場で 1 枚撮る（首振りと目は一続きの動作）。帰りは `see` と
同じ経路——`see_by` を控え、取込で `_see_returned` を立て、調停が出したなら写真つきで
調停が判断し、主LLM が出したなら主LLM へ。見た印も同じ形で残る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.loop.arbiter import Decision as ArbiterDecision
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger
from familiar_agent.poses import Pose
from familiar_agent.tools.camera import CameraTool

from tests.test_event_loop import _agent

POSES = [Pose("押入", -0.254, -0.143), Pose("正面", -0.078, -0.143)]


def _cam(position):
    cam = CameraTool.__new__(CameraTool)
    cam._poses = list(POSES)
    cam.position = AsyncMock(return_value=position)
    cam.move_to = AsyncMock(return_value="ok")
    cam.capture = AsyncMock(return_value=("B64", "/tmp/c.jpg"))
    return cam


def test_look_takes_a_picture_where_it_turned():
    cam = _cam((-0.078, -0.143))
    text, image = asyncio.run(cam.call("look", {"direction": "右"}))
    assert "押入" in text and image == "B64"
    cam.capture.assert_awaited_once()


def test_look_that_could_not_turn_takes_no_picture():
    cam = _cam((-0.254, -0.143))  # 右端
    text, image = asyncio.run(cam.call("look", {"direction": "右"}))
    assert image is None
    cam.capture.assert_not_awaited()


def _ip_with_camera():
    a = _agent(stream_returns=[])
    cam = MagicMock()
    cam.call = AsyncMock(return_value=("右の押入のほうを向いた。", "B64"))
    cam.position = AsyncMock(return_value=(-0.254, -0.143))
    cam.last_capture_path = "/tmp/capture.jpg"
    a._camera = cam
    a._person_detector = MagicMock()
    a._person_detector.labels = AsyncMock(return_value=["closet"])
    a.poses = AsyncMock(return_value=POSES)
    a.config.camera.pose_tolerance = 0.02
    return a, InformationProcessing(a)


def test_the_look_completion_carries_the_seen_mark():
    a, ip = _ip_with_camera()
    with patch("familiar_agent.loop.event_loop.extract_entities", AsyncMock(return_value=[])):
        out = asyncio.run(ip._run_camera("look", {"direction": "右"}))
    assert "押入" in out and "closet" in out


async def _returned_look(ip):
    ip._req.lookups.append(Lookup(index=1, action="look", query="右を見に行く", generation=0))
    ip._drained_completions.append(
        Trigger(kind="完了", query="右を見に行く", result="押入を見た。", index=1)
    )
    await ip._intake()


def test_a_look_thrown_by_the_arbiter_comes_back_to_the_arbiter_with_the_photo():
    async def scenario():
        a, ip = _ip_with_camera()
        ip._req.see_by = "調停"
        ip._seen_image = lambda memories: ("B64", "/tmp/c.jpg")
        await _returned_look(ip)
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="light", text="右を向いたよ")),
        ) as arb:
            d = await ip._decide(
                utterance="右向いて", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return d, arb

    d, arb = asyncio.run(scenario())
    assert arb.called and arb.call_args.kwargs.get("image_b64") == "B64"
    assert d.branch == "light"


def test_a_look_thrown_by_the_main_llm_skips_the_arbiter():
    async def scenario():
        a, ip = _ip_with_camera()
        ip._req.see_by = "主LLM"
        await _returned_look(ip)
        with patch("familiar_agent.loop.event_loop.arbitrate", new=AsyncMock()) as arb:
            d = await ip._decide(
                utterance="右向いて", workspace_ctx="", present_ctx="", capped=False, round_=1
            )
        await ip.close()
        return d, arb

    d, arb = asyncio.run(scenario())
    assert not arb.called and d.branch == "full"


def test_the_arbiters_look_is_marked_as_hers():
    """調停が `look` を出したら `see_by` に控える（`see` と同じ）。"""
    a, ip = _ip_with_camera()
    ip._start_lookup = MagicMock()
    ip._say_filler = AsyncMock()
    from familiar_agent.loop.event_loop import _tool_input_of  # noqa: F401

    decision = ArbiterDecision(
        branch="action", action="look", query="首を向ける", tool_input={"direction": "右"}
    )
    asyncio.run(ip._dispatch_arbiter_action(decision, utterance="右向いて"))
    assert ip._req.see_by == "調停"
