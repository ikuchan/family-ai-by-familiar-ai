"""環-aw 第 7 段：体を状態図の領域にする（`設計方針_パジュの状態` §3.6・決定イ、ウ）。

体の状態は、止まっている（初め）・首を動かしている・見回っている。
- 首を動かしているは事象で動く（声と同じ形）。持ち主はカメラの `move_to` で、動かし始めに `neck_began`、終わりに
  `neck_ended` を送る（上限で諦めても、読めなくても、打ち切られても `finally` で送る）。カメラは状態図を待たない。
- 見回っているは、話しかけない求め（seeking・safety の発火）が開いているあいだ。頭が受けている `quiet_request_opened` で入り、
  `request_closed` で抜ける（新しい持ち主は要らない）。
- 見回っているあいだは、首の事象で移らない（自分へ戻る・図のとおり〔仮（本人の確認待ち）〕）。
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.paju_state import PajuState
from familiar_agent.tools.camera import CameraTool


def _body(sm: PajuState) -> str:
    return sm.body_state()


def test_the_body_starts_still_and_follows_the_neck():
    async def go():
        sm = PajuState()
        await sm.start()
        assert _body(sm) == "body_still" and " 体=止まっている" in sm.describe()
        await sm.fire("neck_began")
        assert _body(sm) == "neck_moving" and " 体=首を動かしている" in sm.describe()
        await sm.fire("neck_ended")
        assert _body(sm) == "body_still"

    asyncio.run(go())


def test_a_quiet_request_is_a_patrol_until_it_closes(caplog):
    async def go():
        sm = PajuState()
        await sm.start()
        await sm.fire("quiet_request_opened")
        assert _body(sm) == "patrolling" and " 体=見回っている" in sm.describe()
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("neck_began")  # 見回りの 1 歩で首が動く
            assert _body(sm) == "patrolling"
            await sm.fire("neck_ended")
        assert not [
            r for r in caplog.records if "捨てた" in r.getMessage()
        ]  # 歩ごとに捨てた行を出さない
        await sm.fire("request_closed")
        assert _body(sm) == "body_still"

    asyncio.run(go())


def test_a_spoken_request_is_not_a_patrol():
    """**反証の側**：話しかける求めは見回りではない。"""

    async def go():
        sm = PajuState()
        await sm.start()
        await sm.fire("request_opened")
        assert _body(sm) == "body_still"

    asyncio.run(go())


def test_a_late_neck_end_after_the_patrol_is_not_dropped(caplog):
    async def go():
        sm = PajuState()
        await sm.start()
        await sm.fire("quiet_request_opened")
        await sm.fire("neck_began")
        await sm.fire("request_closed")
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("neck_ended")
        assert _body(sm) == "body_still"
        assert not [r for r in caplog.records if "捨てた" in r.getMessage()]

    asyncio.run(go())


# ── カメラが送る ─────────────────────────────────────────────────────────────


def _status(x: float, y: float):
    st = MagicMock()
    st.Position.PanTilt.x, st.Position.PanTilt.y = x, y
    return st


def _camera(positions, *, move_fails=False):
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    queue = [_status(*p) for p in positions]

    async def status(_req):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    ptz = MagicMock()
    ptz.AbsoluteMove = AsyncMock(side_effect=RuntimeError("動かない") if move_fails else None)
    ptz.GetStatus = AsyncMock(side_effect=status)
    cam._ptz, cam._profile_token, cam._cam_onvif = ptz, "profile_1", MagicMock()
    cam.SETTLE_STEP_SEC, cam.SETTLE_MAX_SEC, cam.SETTLE_FOLLOW_SEC = 0.01, 0.03, 0.05
    sent: list[str] = []
    cam.on_neck = sent.append
    return cam, sent


def test_the_camera_tells_the_neck_began_and_ended():
    cam, sent = _camera([(0.0, 0.0), (0.5, 0.2)])
    asyncio.run(cam.move_to(0.5, 0.2))
    assert sent == ["neck_began", "neck_ended"]


def test_the_end_is_told_even_when_the_move_gives_up_or_fails():
    cam, sent = _camera([(0.0, 0.0)] * 50)  # いつまでも着かない
    asyncio.run(cam.move_to(0.5, 0.2))
    assert sent == ["neck_began", "neck_ended"]
    cam, sent = _camera([(0.0, 0.0)], move_fails=True)
    asyncio.run(cam.move_to(0.5, 0.2))
    assert sent == ["neck_began", "neck_ended"]


def test_the_agent_sends_neck_events_in_order_without_making_the_camera_wait():
    from familiar_agent.agent import EmbodiedAgent

    async def go():
        sm = PajuState()
        await sm.start()
        agent = SimpleNamespace(paju_state=sm)
        async with sm._turn():  # 状態図の順番がふさがっていても、カメラは待たない
            EmbodiedAgent._on_neck(agent, "neck_began")
            EmbodiedAgent._on_neck(agent, "neck_ended")
            assert _body(sm) == "body_still"
        await asyncio.wait({agent._neck_tail})
        assert _body(sm) == "body_still"  # 始めて終えた（順番どおり）
        EmbodiedAgent._on_neck(agent, "neck_began")
        await asyncio.wait({agent._neck_tail})
        assert _body(sm) == "neck_moving"

    asyncio.run(go())


def test_a_quiet_request_the_head_refused_is_not_a_patrol(caplog):
    """**反証の側**：話しかける求めの最中に届いた quiet_request_opened は、頭が断る。体も見回りにしない（捨てた行は残る）。"""

    async def go():
        sm = PajuState()
        await sm.start()
        await sm.fire("request_opened")
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("quiet_request_opened")
        assert _body(sm) == "body_still"
        assert any(
            "quiet_request_opened を受ける移りが無いので捨てた" in r.getMessage()
            for r in caplog.records
        )

    asyncio.run(go())


def test_overlapping_moves_go_one_after_another():
    """首の動きは 1 本ずつ（見回りの 1 歩と look が重なっても、始めと終わりが入れ替わらない）。"""
    cam, sent = _camera([(0.0, 0.0), (0.5, 0.2), (0.5, 0.2), (-0.3, 0.1)])

    async def go():
        await asyncio.gather(cam.move_to(0.5, 0.2), cam.move_to(-0.3, 0.1))

    asyncio.run(go())
    assert sent == ["neck_began", "neck_ended", "neck_began", "neck_ended"]
