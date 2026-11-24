"""環-aw 第 6 段 項目 3：歌を状態図の口の領域にする（`設計方針_パジュの状態` §3.5・決定オ、ヌ、フ）。

歌は、パジュがかけた音楽の領域で、状態は止まっている（初め）・鳴っている・絞っている。持ち主は音楽の状態
（`agent._music_state` の鳴っている印と、絞っているあいだの音量の基準 `ducked_base`）で、状態図はその写し（知らせの音と
同じ形）。写すのは自律機構の輪（0.5 秒ごと・`copy_song`）だけ。絞る・戻す・鳴っているかの判定は、いままでどおり
音楽の側が持ち主を読む（状態図を待たない）。
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from familiar_agent.core.music_state import MusicState
from familiar_agent.loop.paju_state import PajuState, song_of
from familiar_agent.loop.tonic import Tonic


def test_the_chart_has_a_song_region_that_starts_stopped():
    async def go():
        sm = PajuState()
        await sm.start()
        assert sm.song_state() == "song_stopped"
        assert "歌=止まっている" in sm.describe()
        await sm.fire("song_began")
        assert "歌=鳴っている" in sm.describe()
        await sm.fire("song_quieted")
        assert "歌=絞っている" in sm.describe()
        await sm.fire("song_restored")
        assert sm.song_state() == "song_playing"
        await sm.fire("song_ended")
        assert sm.song_state() == "song_stopped"

    asyncio.run(go())


def test_the_owner_is_read_from_the_music_state():
    assert song_of(None) == "song_stopped"
    assert song_of(MusicState()) == "song_stopped"
    assert song_of(MusicState(playing=True)) == "song_playing"
    assert song_of(MusicState(playing=True, ducked_base=0.4)) == "song_ducked"
    # 止めたあとに絞りの基準だけ残っていても、鳴っていなければ止まっている（家族がかけた曲は歌に入れない・決定フ）
    assert song_of(MusicState(playing=False, ducked_base=0.4)) == "song_stopped"


def test_copy_song_follows_the_owner_and_logs_only_changes(caplog):
    async def go():
        sm = PajuState()
        owner = {"s": "song_playing"}
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.copy_song(lambda: owner["s"])
            await sm.copy_song(lambda: owner["s"])  # 同じなら何もしない
            owner["s"] = "song_ducked"
            await sm.copy_song(lambda: owner["s"])
            owner["s"] = "song_stopped"  # 絞っている最中に止めた
            await sm.copy_song(lambda: owner["s"])
            owner["s"] = "song_ducked"  # 0.5 秒のあいだに鳴らし始めて絞った
            await sm.copy_song(lambda: owner["s"])
        lines = [r.getMessage() for r in caplog.records if "歌を写した" in r.getMessage()]
        assert lines == [
            "状態図 歌を写した：鳴っている",
            "状態図 歌を写した：絞っている",
            "状態図 歌を写した：止まっている",
            "状態図 歌を写した：絞っている",
        ]

    asyncio.run(go())


def test_a_broken_owner_is_warned_and_does_not_raise(caplog):
    def broken() -> str:
        raise RuntimeError("読めない")

    async def go():
        sm = PajuState()
        await sm.start()
        with caplog.at_level(logging.WARNING):
            await sm.copy_song(broken)
        assert any("歌を写せなかった" in r.getMessage() for r in caplog.records)
        assert sm.song_state() == "song_stopped"

    asyncio.run(go())


def test_the_tonic_copies_the_music_state_each_turn():
    async def go():
        sm = PajuState()
        await sm.start()
        state = MusicState()
        agent = SimpleNamespace(paju_state=sm, _music_state=state)
        t = Tonic(SimpleNamespace(), agent=agent)
        state.playing = True
        await t._copy_song()
        assert sm.song_state() == "song_playing"
        state.ducked_base = 0.5
        await t._copy_song()
        assert sm.song_state() == "song_ducked"
        state.ducked_base, state.playing = None, False
        await t._copy_song()
        assert sm.song_state() == "song_stopped"

    asyncio.run(go())
