"""音楽の音量を戻す 10 秒を、声の処理の中で待たない（出-bd ①・2026-10-10・本人の決定）。

音楽が鳴っているあいだは声のあいだだけ絞り、声が終わって 10 秒おいて戻す。その 10 秒を `duck_while_speaking` の
中で待っていたので、声を 1 つ出すたびに次へ進めなかった（実機 10/10 11:05:57：17 字の聞き返しで `DIF 声 12.46 秒`）。
戻しは裏で予約し、すぐ返す。予約中に次の声が来たら予約を取り消し、絞ったまま最初に読んだ基準を引き継ぐ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.music_rules import DUCK_RATIO
from familiar_agent.core.music_state import MusicState
from familiar_agent.loop.music_watch import duck_while_speaking


def _io(volume: float = 0.8):
    io = MagicMock()
    io.status = AsyncMock(return_value={"playing": True, "volume": volume})
    io.set_volume = AsyncMock(return_value=True)
    return io


async def _speak():
    return "話した"


@pytest.mark.asyncio
async def test_it_returns_before_the_music_comes_back():
    io = _io()
    state = MusicState(playing=True)
    release = asyncio.Event()

    async def sleep(_sec):
        await release.wait()

    got = await asyncio.wait_for(
        duck_while_speaking(io=io, bus=MagicMock(), state=state, speak=_speak, sleep=sleep),
        timeout=1.0,
    )
    assert got == "話した"
    assert [c.args[1] for c in io.set_volume.await_args_list] == [pytest.approx(0.8 * DUCK_RATIO)]
    release.set()
    await state.restore_task
    assert io.set_volume.await_args_list[-1].args[1] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_a_second_voice_keeps_it_down_and_the_first_base():
    io = _io()
    state = MusicState(playing=True)
    release = asyncio.Event()

    async def sleep(_sec):
        await release.wait()

    async def once():
        await asyncio.wait_for(
            duck_while_speaking(io=io, bus=MagicMock(), state=state, speak=_speak, sleep=sleep),
            timeout=1.0,
        )

    await once()
    first = state.restore_task
    io.status = AsyncMock(return_value={"playing": True, "volume": 0.2})  # 絞ったいまの音量
    await once()
    await asyncio.sleep(0)
    assert first.cancelled()
    io.status.assert_not_awaited()  # 読み直さない（読めば 0.2 を基準にしてしまう）
    release.set()
    await state.restore_task
    volumes = [c.args[1] for c in io.set_volume.await_args_list]
    assert volumes == [pytest.approx(0.8 * DUCK_RATIO), pytest.approx(0.8)]


@pytest.mark.asyncio
async def test_the_reply_after_a_cue_does_not_wait_for_the_music():
    from familiar_agent.io.dif import DIF

    io = _io()
    state = MusicState(playing=True)
    release = asyncio.Event()

    async def sleep(_sec):
        await release.wait()

    async def ducker(speak):
        return await duck_while_speaking(
            io=io, bus=MagicMock(), state=state, speak=speak, sleep=sleep
        )

    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: ok", None))
    dif = DIF(tts=tts)
    dif.set_music_ducker(ducker)
    dif._cue_task = asyncio.ensure_future(ducker(_speak))  # 合図が鳴った
    await asyncio.wait_for(dif.speak("こんにちは"), timeout=1.0)
    tts.call.assert_awaited_once()
    release.set()
    await state.restore_task
