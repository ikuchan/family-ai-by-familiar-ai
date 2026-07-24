"""ONVIF セッション後始末：move 失敗時に aiohttp セッションを閉じてから破棄する。

閉じずに `_cam_onvif=None` すると `Unclosed client session` が漏れ、繰り返すと qasync
イベントループの aiohttp 処理が劣化して PTZ が starve する。失敗経路で close を保証する。
"""

from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock

from familiar_agent.tools.camera import CameraTool


class _FakeCam:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _tool_with_connected(fake_cam, ptz) -> CameraTool:
    tool = CameraTool.__new__(CameraTool)  # __init__ を回避して必要属性だけ用意
    tool._cam_onvif = fake_cam  # not None → _ensure_connected は即 True
    tool._ptz = ptz
    tool._profile_token = "Profile_1"
    return tool


def test_move_closes_onvif_session_on_failure():
    fake = _FakeCam()
    ptz = AsyncMock()
    ptz.RelativeMove = AsyncMock(side_effect=RuntimeError("boom"))
    tool = _tool_with_connected(fake, ptz)

    res = asyncio.run(tool.move("left"))

    assert fake.closed is True          # 失敗時にセッションを閉じた
    assert tool._cam_onvif is None       # そのうえで破棄
    assert "failed" in res.lower()


def test_move_success_keeps_session():
    fake = _FakeCam()
    ptz = AsyncMock()
    ptz.RelativeMove = AsyncMock(return_value=None)
    tool = _tool_with_connected(fake, ptz)

    res = asyncio.run(tool.move("right"))

    assert fake.closed is False          # 成功時は閉じない（再利用）
    assert tool._cam_onvif is fake
    assert "looked" in res.lower()
