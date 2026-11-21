"""環-aw 第 6 段 項目 2：歌は、声のあいだだけでなく、知らせの音（タイマー、アラーム）が鳴っているあいだも絞る（決定ヌ）。

声と知らせの音は時間が重なりうる（再生の鍵で交互に鳴る）。絞っている最中（`ducked_base` がある）なら 2 度は絞らない。
戻す仕事は、戻す前に持ち主（`busy`：知らせの音を鳴らしている仕事、口の声）を読み、忙しければ待ち直す（禁止の条件）。
戻す値は、そのときの `ducked_base`（絞っているあいだに人が変えた値）。数える印は持たない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.music_rules import DUCK_RATIO
from familiar_agent.core.music_state import MusicState
from familiar_agent.io import dif as dif_mod
from familiar_agent.loop.music_watch import duck_while_speaking


def _io(volume: float = 0.8):
    io = MagicMock()
    vol = {"v": volume}

    async def status(_bus):
        await asyncio.sleep(0)  # 読むのに一拍かかる（同時に入った者が追いつける）
        return {"playing": True, "volume": vol["v"]}

    async def set_volume(_bus, v):
        await asyncio.sleep(0)
        vol["v"] = v
        return True

    io.status = AsyncMock(side_effect=status)
    io.set_volume = AsyncMock(side_effect=set_volume)
    io.vol = vol
    return io


class _Sleep:
    """戻しの待ちを、試験が 1 回ずつ進める。"""

    def __init__(self) -> None:
        self.tick = asyncio.Event()
        self.waits = 0

    async def __call__(self, _sec: float) -> None:
        self.waits += 1
        await self.tick.wait()
        self.tick.clear()


async def _settle() -> None:
    for _ in range(30):
        await asyncio.sleep(0)


def _duck(io, state, sleep, busy=None):
    def run(play):
        return duck_while_speaking(
            io=io, bus=MagicMock(), state=state, speak=play, sleep=sleep, busy=busy
        )

    return run


@pytest.mark.asyncio
async def test_two_entering_at_once_duck_only_once():
    io = _io()
    state = MusicState(playing=True)
    go = asyncio.Event()

    async def hold():
        await go.wait()

    duck = _duck(io, state, _Sleep())
    a = asyncio.ensure_future(duck(hold))
    b = asyncio.ensure_future(duck(hold))  # 待ちを挟まずに 2 つめ
    await _settle()
    assert io.vol["v"] == pytest.approx(0.8 * DUCK_RATIO)
    assert io.set_volume.await_count == 1
    go.set()
    await asyncio.gather(a, b)
    state.restore_task.cancel()


@pytest.mark.asyncio
async def test_a_voice_ending_mid_ring_does_not_bring_the_music_back():
    io = _io()
    state = MusicState(playing=True)
    sleep = _Sleep()
    ringing = {"on": True}
    duck = _duck(io, state, sleep, busy=lambda: ringing["on"])
    ring_go = asyncio.Event()

    async def ring():
        await ring_go.wait()

    async def voice():
        return "話した"

    t_ring = asyncio.ensure_future(duck(ring))
    await _settle()
    await duck(voice)  # 鳴っている途中で声が終わる
    sleep.tick.set()
    await _settle()
    assert io.vol["v"] == pytest.approx(0.8 * DUCK_RATIO)  # まだ鳴っているので戻さない
    ringing["on"] = False
    ring_go.set()
    await t_ring
    sleep.tick.set()
    await _settle()
    assert io.vol["v"] == pytest.approx(0.8)
    assert state.ducked_base is None


@pytest.mark.asyncio
async def test_a_volume_changed_while_ducked_is_where_it_comes_back_to():
    io = _io()
    state = MusicState(playing=True)
    sleep = _Sleep()

    async def voice():
        state.ducked_base = (
            0.9  # 絞っているあいだに「音量を上げて」（tools/music.py が基準を書き換える）
        )
        return "話した"

    await _duck(io, state, sleep)(voice)
    sleep.tick.set()
    await _settle()
    assert io.vol["v"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_stopped_while_ducking_still_restores():
    io = _io()
    state = MusicState(playing=True)
    sleep = _Sleep()
    entered = asyncio.Event()
    real_set = io.set_volume.side_effect

    async def slow_set(bus, v):
        entered.set()
        await asyncio.sleep(10)
        await real_set(bus, v)

    io.set_volume.side_effect = slow_set

    async def ring():
        await asyncio.sleep(10)

    t = asyncio.ensure_future(_duck(io, state, sleep)(ring))
    await entered.wait()
    t.cancel()  # 絞る途中で止めた（stop_ring）
    with pytest.raises(asyncio.CancelledError):
        await t
    assert state.restore_task is not None and not state.restore_task.done()
    state.restore_task.cancel()


def test_the_alarm_ring_sounds_inside_the_ducker(monkeypatch):
    played: list[str] = []

    async def _play(path, gain):  # noqa: ANN001
        played.append("鳴った")
        await asyncio.sleep(0)
        return False  # 1 回で終わる

    monkeypatch.setattr(dif_mod, "_play_wav", _play)
    seen: list[tuple[str, list[str]]] = []

    async def ducker(play, why="話す"):
        before = list(played)
        out = await play()
        seen.append((why, before))
        return out

    async def go():
        d = dif_mod.DIF()
        d.set_music_ducker(ducker)
        d.ring(seconds=30)
        await _settle()

    asyncio.run(go())
    assert seen == [("知らせの音が鳴る", [])]  # 絞ってから鳴らした
    assert played == ["鳴った"]


@pytest.mark.asyncio
async def test_a_failed_duck_does_not_look_ducked():
    io = _io()
    io.set_volume.side_effect = RuntimeError("DBus が落ちた")
    state = MusicState(playing=True)

    async def voice():
        return "話した"

    assert await _duck(io, state, _Sleep())(voice) == "話した"  # 声は出す
    assert state.ducked_base is None


@pytest.mark.asyncio
async def test_a_busy_that_never_ends_still_restores_in_the_end(monkeypatch, caplog):
    from familiar_agent.loop import music_watch

    monkeypatch.setattr(music_watch, "_MAX_REWAITS", 3)
    io = _io()
    state = MusicState(playing=True)

    async def no_wait(_sec):
        await asyncio.sleep(0)

    async def voice():
        return "話した"

    await _duck(io, state, no_wait, busy=lambda: True)(voice)
    await state.restore_task
    assert io.vol["v"] == pytest.approx(0.8)
    assert any("静かにならない" in r.getMessage() for r in caplog.records)
