"""音楽のふだんの音量を覚え、次にかけるときに使う（知-bi・2026-10-17・本人の決定イ・ア・ウ）。

人が声で変えたときと、止めたとき（「止めて」・30 分の寿命・鳴り止んだ）の音量を `agent_state` の
`music_volume` に記録し、かけ始めにその音量にする。声で頼むと合図の声で先に絞られるので、記録するのは
**絞る前の値**（`ducked_base`）である。19:42 に、話すあいだ 4 分の 1 に絞って止まったように聞こえた。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_state
from familiar_agent.core.music_state import MusicState
from familiar_agent.tools.music import MusicTool

TABLE = (("ケイマン", "spotify:playlist:aaa", False),)


@pytest.fixture
def saved(monkeypatch):
    """DB の代わり。`box["v"]` が記録された音量（無ければ None）。"""
    box: dict = {"v": None, "writes": []}
    monkeypatch.setattr(music_state, "saved_volume", lambda: box["v"])

    def save(v):
        box["v"] = v
        box["writes"].append(v)
        return True

    monkeypatch.setattr(music_state, "save_volume", save)
    return box


def _tool(state: MusicState, *, volume: float = 0.5, playing: bool = True):
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.stop = AsyncMock(return_value=True)
    io.set_volume = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    io.next_track = AsyncMock(return_value=True)
    io.status = AsyncMock(return_value={"playing": playing, "volume": volume})
    return MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: TABLE, state=state), io


def _volumes(io) -> list[float]:
    return [c.args[1] for c in io.set_volume.await_args_list]


# ── かけ始め ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_playing_starts_at_the_remembered_volume(saved):
    saved["v"] = 0.3
    tool, io = _tool(MusicState())
    _, ok = await tool.call("play_music", {"name": "ケイマン"})
    assert ok
    assert _volumes(io) == [pytest.approx(0.3)]


@pytest.mark.asyncio
async def test_without_a_record_the_volume_is_left_alone(saved):
    """**反証の側**：記録が無ければ音量に触らない（spotifyd の初期値のまま）。"""
    tool, io = _tool(MusicState())
    await tool.call("play_music", {"name": "ケイマン"})
    io.set_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_while_ducked_only_the_way_back_is_changed(saved):
    """話している最中にかけ直したら、いまは上げず、戻す先だけを記録した値にする。"""
    saved["v"] = 0.3
    state = MusicState(playing=True, ducked_base=0.5)
    tool, io = _tool(state, volume=0.25)
    await tool.call("play_music", {"name": "ケイマン"})
    io.set_volume.assert_not_awaited()
    assert state.ducked_base == pytest.approx(0.3)


# ── 声で変える ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_louder_while_ducked_counts_from_before_the_duck(saved):
    """合図の声で絞られていても、絞る前の値に足す。絞った値（0.25）に足すと 0.40 になっていた。"""
    state = MusicState(playing=True, ducked_base=0.5)
    tool, io = _tool(state, volume=0.25)
    out, ok = await tool.call("music_volume", {"how": "大きく"})
    assert ok and "65%" in out
    assert saved["writes"] == [pytest.approx(0.65)]
    assert state.ducked_base == pytest.approx(0.65)  # 戻す先も新しい値
    assert _volumes(io) == [pytest.approx(0.65 * 0.5)]  # いまは絞ったまま


@pytest.mark.asyncio
async def test_a_softer_without_a_duck_counts_from_now(saved):
    """**反証の側**：絞っていなければ、いまの値から引き、その値にする。"""
    tool, io = _tool(MusicState(playing=True), volume=0.5)
    await tool.call("music_volume", {"how": "小さく"})
    assert saved["writes"] == [pytest.approx(0.35)]
    assert _volumes(io) == [pytest.approx(0.35)]


# ── 止めたとき ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stopping_remembers_the_volume_before_the_duck(saved):
    state = MusicState(playing=True, ducked_base=0.4)
    tool, _ = _tool(state, volume=0.2)
    _, ok = await tool.call("stop_music", {})
    assert ok
    assert saved["writes"] == [pytest.approx(0.4)]


@pytest.mark.asyncio
async def test_a_failed_stop_remembers_nothing(saved):
    """**反証の側**：止められなかったら記録しない。"""
    tool, io = _tool(MusicState(playing=True), volume=0.4)
    io.stop = AsyncMock(return_value=False)
    await tool.call("stop_music", {})
    assert saved["writes"] == []


@pytest.mark.asyncio
async def test_the_thirty_minute_stop_remembers_the_volume(saved):
    from familiar_agent.core.music_rules import MUSIC_MAX_SEC
    from familiar_agent.loop.music_watch import check_music_expired

    io = MagicMock()
    io.stop = AsyncMock(return_value=True)
    io.status = AsyncMock(return_value={"playing": True, "volume": 0.45})
    state = MusicState(playing=True, started_at=1000.0)
    await check_music_expired(
        io=io, bus=MagicMock(), state=state, now=1000.0 + MUSIC_MAX_SEC, say=lambda _t: None
    )
    assert saved["writes"] == [pytest.approx(0.45)]


@pytest.mark.asyncio
async def test_a_found_stop_remembers_the_volume_it_read(saved):
    from familiar_agent.loop.music_watch import observe

    io = MagicMock()
    io.status = AsyncMock(return_value={"playing": False, "volume": 0.6})
    await observe(io=io, bus=MagicMock(), state=MusicState(playing=True), record=lambda _t: None)
    assert saved["writes"] == [pytest.approx(0.6)]


@pytest.mark.asyncio
async def test_an_unreadable_stop_remembers_nothing(saved):
    """**反証の側**：鳴り止んだが音量が読めない（スマホから止めて spotifyd が居ない）なら記録しない。"""
    from familiar_agent.loop.music_watch import observe

    io = MagicMock()
    io.status = AsyncMock(return_value={})
    await observe(io=io, bus=MagicMock(), state=MusicState(playing=True), record=lambda _t: None)
    assert saved["writes"] == []


# ── DB の口（テスト DB・試験ごとに `agent_state` は空になる）────────────────────


def test_the_volume_is_written_and_read_back():
    assert music_state.saved_volume() is None  # 記録が無い
    assert music_state.save_volume(0.35)
    assert music_state.saved_volume() == pytest.approx(0.35)
    music_state.save_volume(1.7)  # 範囲の外は 0〜1 に収める
    assert music_state.saved_volume() == pytest.approx(1.0)
