"""首が着くまでの時間をログに残す（知-ay・2026-10-11・本人の決定ア）。

テレビの定点（傾き 0.485）へ向かうと、3 秒の待ちで諦めたときの傾きがほとんど毎回、前の定点の 0.143 のままだった。
横を動かしてから傾きを動かすように見え、遠い定点へは 3 秒で着かない。上限を決め直すため、着くまでの秒数を残す。
諦めたあとも裏で 15 秒まで読み続ける（撮る処理は待たせない）。次の動きが来たら読み続けは取り消す。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.tools.camera import CameraTool


def _status(x: float, y: float):
    st = MagicMock()
    st.Position.PanTilt.x, st.Position.PanTilt.y = x, y
    return st


def _tool(positions: "list[tuple[float, float]]"):
    """`GetStatus` が順に位置を返す（使い切ったら最後の位置のまま）。刻みは短くする。"""
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    queue = [_status(*p) for p in positions]

    async def status(_req):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    ptz = MagicMock()
    ptz.AbsoluteMove = AsyncMock(return_value=None)
    ptz.GetStatus = AsyncMock(side_effect=status)
    cam._ptz = ptz
    cam._profile_token = "profile_1"
    cam._cam_onvif = MagicMock()
    cam.SETTLE_STEP_SEC = 0.01
    cam.SETTLE_MAX_SEC = 0.03
    cam.SETTLE_FOLLOW_SEC = 0.15
    return cam


def _messages(caplog) -> "list[str]":
    return [r.getMessage() for r in caplog.records if r.name == "familiar_agent.tools.camera"]


def test_a_settled_move_logs_the_seconds_and_where_from(caplog):
    caplog.set_level(logging.INFO, logger="familiar_agent.tools.camera")
    cam = _tool([(0.194, -0.143), (0.3, -0.143), (0.491, 0.485)])
    asyncio.run(cam.move_to(0.491, 0.485))
    got = [m for m in _messages(caplog) if "着いた" in m]
    assert len(got) == 1
    assert "秒" in got[0] and "(0.194, -0.143)" in got[0] and "(0.491, 0.485)" in got[0]


def test_a_late_arrival_is_logged_without_holding_the_move(caplog):
    caplog.set_level(logging.INFO, logger="familiar_agent.tools.camera")
    # 出発点 → 3 回は途中（ここで諦める）→ そのあと着く
    cam = _tool([(0.194, 0.143)] + [(0.491, 0.143)] * 5 + [(0.491, 0.485)])

    async def run():
        await cam.move_to(0.491, 0.485)
        returned_before_arrival = not any("遅れて着いた" in m for m in _messages(caplog))
        await asyncio.sleep(0.3)
        return returned_before_arrival

    assert asyncio.run(run())
    msgs = _messages(caplog)
    assert any("着かない" in m for m in msgs)
    late = [m for m in msgs if "遅れて着いた" in m]
    assert len(late) == 1 and "(0.194, 0.143)" in late[0]


def test_no_arrival_within_the_window_is_warned(caplog):
    caplog.set_level(logging.INFO, logger="familiar_agent.tools.camera")
    cam = _tool([(0.194, 0.143), (0.491, 0.143)])

    async def run():
        await cam.move_to(0.491, 0.485)
        await asyncio.sleep(0.3)

    asyncio.run(run())
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "秒でも着かない" in r.getMessage()
    ]
    assert len(warned) == 1


def test_the_next_move_cancels_the_follow_up(caplog):
    caplog.set_level(logging.INFO, logger="familiar_agent.tools.camera")
    cam = _tool([(0.194, 0.143), (0.491, 0.143)])

    async def run():
        await cam.move_to(0.491, 0.485)
        follow = cam._follow
        assert follow is not None and not follow.done()
        await cam.move_to(0.491, 0.143)  # いまの位置へ（すぐ着く）
        await asyncio.sleep(0)
        return follow

    follow = asyncio.run(run())
    assert follow.cancelled()
    assert not any("秒でも着かない" in m for m in _messages(caplog))
