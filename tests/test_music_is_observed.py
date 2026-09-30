"""鳴っているあいだの様子を読み、曲送りと鳴り終わりを O に残す（知-aa 段 2・2026-09-30）。

MPRIS を読むのは `music_watch.observe` の 1 か所で、T の見張り（30 秒ごと）と反復が同じものを呼ぶ。

- **曲送り**：最後に O に書いた曲名（`MusicState.last_title`）と違えば「音楽：曲／アーティスト」を書く。
  黙って聴いているあいだの曲も残す（本人の決定イ）。プレイリスト名は MPRIS が返さないので書かない。
- **鳴り終わり**：「鳴っている」の印が立っているのに止まっていれば、印を下ろし（声の入口の門が開く）、
  「音楽が止まった」を書く。声には出さず、30 分の一言も言わない。
- 読めない（例外・口が無い）ときは何もしない。機器は落ちる前提で、読めないことを「止まった」とは見ない。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.music_rules import MUSIC_MAX_SEC
from familiar_agent.core.music_state import MusicState
from familiar_agent.loop.music_watch import observe


def _io(status):
    io = MagicMock()
    if isinstance(status, Exception):
        io.status = AsyncMock(side_effect=status)
    else:
        io.status = AsyncMock(return_value=status)
    io.stop = AsyncMock(return_value=True)
    return io


PLAYING = {"playing": True, "title": "海へ", "artist": "ケイマン", "volume": 0.5}


@pytest.mark.asyncio
async def test_a_new_song_is_recorded_once():
    state = MusicState(playing=True, started_at=1.0)
    recorded: list[str] = []
    got = await observe(io=_io(PLAYING), bus=MagicMock(), state=state, record=recorded.append)
    assert recorded == ["音楽：海へ／ケイマン"]
    assert state.last_title == "海へ"
    assert got == PLAYING
    await observe(io=_io(PLAYING), bus=MagicMock(), state=state, record=recorded.append)
    assert recorded == ["音楽：海へ／ケイマン"]  # 同じ曲は書かない


@pytest.mark.asyncio
async def test_when_it_stopped_elsewhere_the_mark_comes_down():
    state = MusicState(playing=True, started_at=1.0, last_title="海へ")
    recorded: list[str] = []
    await observe(
        io=_io({"playing": False, "title": "海へ", "artist": "", "volume": 0.5}),
        bus=MagicMock(),
        state=state,
        record=recorded.append,
    )
    assert state.playing is False
    assert recorded == ["音楽が止まった"]


@pytest.mark.asyncio
async def test_no_player_means_it_stopped_too():
    state = MusicState(playing=True, started_at=1.0)
    recorded: list[str] = []
    await observe(io=_io({}), bus=MagicMock(), state=state, record=recorded.append)
    assert state.playing is False and recorded == ["音楽が止まった"]


@pytest.mark.asyncio
async def test_a_read_that_fails_changes_nothing():
    state = MusicState(playing=True, started_at=1.0, last_title="海へ")
    recorded: list[str] = []
    got = await observe(
        io=_io(RuntimeError("dbus")), bus=MagicMock(), state=state, record=recorded.append
    )
    assert got == {}
    assert state.playing is True and recorded == []


@pytest.mark.asyncio
async def test_nothing_is_read_while_nothing_plays():
    io = _io(PLAYING)
    state = MusicState()
    recorded: list[str] = []
    assert await observe(io=io, bus=MagicMock(), state=state, record=recorded.append) == {}
    io.status.assert_not_awaited()
    assert recorded == []


def test_starting_to_play_forgets_the_last_song():
    """かけ始めで控えを空にする。同じプレイリストをかけ直したとき、最初の曲も書く。"""
    from familiar_agent.tools.music import MusicTool

    state = MusicState(last_title="海へ")
    tool = MusicTool(io=MagicMock(), bus=MagicMock(), table=lambda: (), state=state)
    tool._mark(True)
    assert state.last_title == ""


# ── T の見張り ─────────────────────────────────────────────────────────────


def _tonic(status, *, started_at):
    from familiar_agent.loop.tonic import Tonic

    t = MagicMock(spec=Tonic)
    tool = MagicMock()
    tool._io = _io(status)
    tool._bus = MagicMock(return_value=MagicMock())
    t._agent = MagicMock()
    t._agent._music_tool = tool
    t._agent._music_state = MusicState(playing=True, started_at=started_at)
    t._dif = MagicMock()
    return t


@pytest.mark.asyncio
async def test_the_tonic_watch_records_what_it_reads(monkeypatch):
    from familiar_agent.loop.tonic import Tonic

    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: 100.0)
    t = _tonic(PLAYING, started_at=90.0)
    await Tonic._check_music(t)
    t._dif.record.assert_called_once_with("音楽", "音楽：海へ／ケイマン")
    t._dif.device.assert_not_called()


@pytest.mark.asyncio
async def test_a_finished_playlist_is_not_announced_at_thirty_minutes(monkeypatch):
    """鳴り終わっていたら、30 分たっても「止めるね」とは言わない（もう鳴っていない）。"""
    from familiar_agent.loop.tonic import Tonic

    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: 1000.0 + MUSIC_MAX_SEC)
    t = _tonic({"playing": False, "title": "", "artist": "", "volume": 0.5}, started_at=1000.0)
    await Tonic._check_music(t)
    t._dif.record.assert_called_once_with("音楽", "音楽が止まった")
    t._dif.device.assert_not_called()
    assert t._agent._music_state.playing is False
