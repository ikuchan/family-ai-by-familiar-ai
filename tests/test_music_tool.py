"""音楽の道具（知-aa 段 1・2026-09-21）。

主LLM が呼ぶ 4 本——`play_music`（かける）・`stop_music`（止める）・`next_track`（次へ）・
`music_volume`（音量）。鳴らす先は MPRIS（`io/music`）で、**状態は溜めず読む**。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.tools.music import MusicTool

TABLE = (("ケイマン", "spotify:playlist:aaa", False), ("ドライブ", "spotify:playlist:bbb", True))


def _tool(**kw):
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.stop = AsyncMock(return_value=True)
    io.next_track = AsyncMock(return_value=True)
    io.set_volume = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    io.status = AsyncMock(
        return_value={"playing": True, "title": "海の歌", "artist": "誰か", "volume": 0.5}
    )
    tool = MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: TABLE, **kw)
    return tool, io


def test_the_four_tools_are_offered():
    tool, _ = _tool()
    assert [d["name"] for d in tool.get_tool_definitions()] == [
        "play_music",
        "stop_music",
        "next_track",
        "music_volume",
    ]
    for d in tool.get_tool_definitions():
        assert d["description"], d["name"]


@pytest.mark.asyncio
async def test_playing_by_name_uses_the_table():
    tool, io = _tool()
    out, ok = await tool.call("play_music", {"name": "ケイマン"})
    assert ok and "ケイマン" in out
    io.play.assert_awaited_once()
    assert io.play.await_args.args[1] == "spotify:playlist:aaa"


@pytest.mark.asyncio
async def test_a_name_that_is_not_in_the_table_is_refused():
    tool, io = _tool()
    out, ok = await tool.call("play_music", {"name": "知らない曲"})
    assert not ok and "知らない曲" in out
    io.play.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_order_follows_the_table_then_the_words():
    tool, io = _tool()
    await tool.call("play_music", {"name": "ドライブ"})  # 表の既定がランダム
    assert io.set_shuffle.await_args.args[1] is True
    io.set_shuffle.reset_mock()
    await tool.call("play_music", {"name": "ドライブ", "order": "順番"})  # 言葉で上書き
    assert io.set_shuffle.await_args.args[1] is False


@pytest.mark.asyncio
async def test_stopping_when_nothing_plays_says_so():
    tool, io = _tool()
    io.stop = AsyncMock(return_value=False)
    out, ok = await tool.call("stop_music", {})
    assert not ok and "鳴っていない" in out


@pytest.mark.asyncio
async def test_the_volume_is_named_in_words():
    tool, io = _tool()
    await tool.call("music_volume", {"how": "大きく"})
    assert io.set_volume.await_args.args[1] > 0.5
    await tool.call("music_volume", {"how": "小さく"})
    assert io.set_volume.await_args.args[1] < 0.5


@pytest.mark.asyncio
async def test_the_frame_says_what_is_playing():
    tool, _ = _tool()
    assert "海の歌" in await tool.frame()


@pytest.mark.asyncio
async def test_no_frame_when_silent():
    tool, io = _tool()
    io.status = AsyncMock(return_value={})
    assert await tool.frame() == ""
