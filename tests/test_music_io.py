"""音楽の口（知-aa 段 1・2026-09-21）。MPRIS（D-Bus）で `spotifyd` を操る。

**バス名は起動のたびに変わる**（`org.mpris.MediaPlayer2.spotifyd.instance<PID>`）ので、前方一致で
探す。**MPRIS はこの機が再生中の機器になってから現れる**ので、無いときは `spotifyd` 自身の口
（`rs.spotifyd.instance…` の `TransferPlayback`）で再生をこちらへ移してから鳴らす（実機で確かめた
2026-09-21：一度も鳴らしていない状態では `org.mpris.MediaPlayer2.*` が出ていない）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.io import music


def _bus(names: "list[str]", player: MagicMock | None = None) -> MagicMock:
    bus = MagicMock()
    bus.list_names = AsyncMock(return_value=names)
    bus.player = AsyncMock(return_value=player)
    return bus


@pytest.mark.asyncio
async def test_the_bus_name_is_found_by_its_prefix():
    names = [
        "org.freedesktop.DBus",
        "org.mpris.MediaPlayer2.spotifyd.instance42",
        "rs.spotifyd.instance42",
    ]
    assert await music.find_player_name(_bus(names)) == "org.mpris.MediaPlayer2.spotifyd.instance42"


@pytest.mark.asyncio
async def test_no_player_yet_is_not_an_error():
    assert await music.find_player_name(_bus(["rs.spotifyd.instance42"])) is None


@pytest.mark.asyncio
async def test_playing_a_uri_goes_through_open_uri():
    player = MagicMock()
    player.call_open_uri = AsyncMock()
    bus = _bus(["org.mpris.MediaPlayer2.spotifyd.instance42"], player)
    assert await music.play(bus, "spotify:playlist:aaa") is True
    player.call_open_uri.assert_awaited_once_with("spotify:playlist:aaa")


@pytest.mark.asyncio
async def test_without_a_player_the_playback_is_transferred_first():
    """一度も鳴らしていないと MPRIS が無い。`TransferPlayback` で移してから鳴らす。"""
    transferred = MagicMock()
    transferred.call_transfer_playback = AsyncMock()
    player = MagicMock()
    player.call_open_uri = AsyncMock()
    bus = MagicMock()
    bus.list_names = AsyncMock(
        side_effect=[["rs.spotifyd.instance42"], ["org.mpris.MediaPlayer2.spotifyd.instance42"]]
    )
    bus.controls = AsyncMock(return_value=transferred)
    bus.player = AsyncMock(return_value=player)
    assert await music.play(bus, "spotify:playlist:aaa") is True
    transferred.call_transfer_playback.assert_awaited_once()
    player.call_open_uri.assert_awaited_once_with("spotify:playlist:aaa")


@pytest.mark.asyncio
async def test_nothing_to_stop_is_said_plainly():
    bus = _bus([])
    assert await music.stop(bus) is False
    assert await music.status(bus) == {}


@pytest.mark.asyncio
async def test_the_status_is_read_not_kept():
    player = MagicMock()
    player.get_playback_status = AsyncMock(return_value="Playing")
    player.get_metadata = AsyncMock(
        return_value={
            "xesam:title": MagicMock(value="海の歌"),
            "xesam:artist": MagicMock(value=["back number"]),
        }
    )
    player.get_volume = AsyncMock(return_value=0.5)
    bus = _bus(["org.mpris.MediaPlayer2.spotifyd.instance42"], player)
    got = await music.status(bus)
    assert got["playing"] is True
    assert got["title"] == "海の歌"
    assert got["artist"] == "back number"
    assert got["volume"] == 0.5


@pytest.mark.asyncio
async def test_the_volume_is_written_within_range():
    player = MagicMock()
    player.set_volume = AsyncMock()
    bus = _bus(["org.mpris.MediaPlayer2.spotifyd.instance42"], player)
    assert await music.set_volume(bus, 1.5) is True
    player.set_volume.assert_awaited_once_with(1.0)
