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


# ── 主LLM へ `[音楽]` の枠を渡す（段 2）──────────────────────────────────────


def _agent_playing(status):
    from familiar_agent.backends import ToolCall
    from familiar_agent.tools.music import MusicTool
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "海へです"})])])
    a._music_state = MusicState(playing=True, started_at=1.0)
    a._music_tool = MusicTool(
        io=_io(status), bus=MagicMock(), table=lambda: (), state=a._music_state
    )
    return a


def test_the_main_llm_is_told_what_is_playing():
    from tests.test_event_loop import _run

    a = _agent_playing(PLAYING)
    _run(a, utterance="パジュ、いま何の曲？")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[音楽] 海へ／ケイマン（音量 50%）" in system


def test_the_frame_says_nothing_plays_when_it_stopped():
    """鳴っていないことも書く（知-ak 段 5・2026-10-07 実機 18:43）。書かないと、記憶の「かけ始めた」がいまの
    状態のように読まれ、`play_music` を呼ばずに「もうかけてますよ」と答えた。"""
    from tests.test_event_loop import _run

    a = _agent_playing({"playing": False, "title": "", "artist": "", "volume": 0.5})
    _run(a, utterance="パジュ、いま何の曲？")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[音楽] いまは何も鳴っていない" in system


def test_the_frame_says_nothing_plays_when_paju_has_not_started_it():
    from tests.test_event_loop import _run

    a = _agent_playing(PLAYING)
    a._music_state.playing = False  # パジュはかけていない（読まない）
    _run(a, utterance="パジュ、ケイマンかけて")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[音楽] いまは何も鳴っていない" in system


@pytest.mark.asyncio
async def test_no_frame_when_the_state_cannot_be_read():
    """読めないときは書かない（分からないのに「鳴っていない」とは言わない）。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    a = _agent_playing(PLAYING)
    a._music_tool._io.status = AsyncMock(side_effect=RuntimeError("D-Bus"))
    ip = InformationProcessing(a)
    ip.note_device = MagicMock()  # type: ignore[method-assign]
    frame = await ip._music_now()
    await ip.close()
    assert frame == ""


def test_no_frame_without_the_music_tool():
    from tests.test_event_loop import _agent, _run

    a = _agent(stream_returns=[])
    a._music_tool = None
    _run(a, utterance="パジュ、いま何時？")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"] or [])
    assert "[音楽]" not in system


@pytest.mark.asyncio
async def test_the_iteration_records_a_song_change_too():
    from familiar_agent.loop.event_loop import InformationProcessing

    a = _agent_playing(PLAYING)
    ip = InformationProcessing(a)
    ip.note_device = MagicMock()  # type: ignore[method-assign]
    frame = await ip._music_now()
    await ip.close()
    assert frame.startswith("[音楽] 海へ")
    ip.note_device.assert_called_once_with("音楽", "音楽：海へ／ケイマン")
