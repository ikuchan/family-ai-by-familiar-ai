"""音楽の結線（知-aa 段 1・2026-09-21）。門・寿命・減音。

鳴っているあいだは**音楽の話だけ**を通す（時間の道具は通さない・本人の決定）。30 分で止めて
一言言う。パジュが声を出しているあいだ（と、その後 10 秒）は 4 分の 1 に絞る。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.music_rules import DUCK_RATIO, MUSIC_MAX_SEC


# ── 門（聞かないあいだと同じ器）─────────────────────────────────────────────


def test_the_gate_passes_only_music_words_while_playing():
    from familiar_agent.core.music_rules import is_music_word

    table = (("ケイマン", "spotify:playlist:aaa", False),)
    assert is_music_word("止めて", table)
    assert is_music_word("次の曲", table)
    assert is_music_word("ケイマンかけて", table)
    assert not is_music_word("3 分測って", table)  # 時間の道具は通さない
    assert not is_music_word("今日の天気は？", table)


def test_the_agent_reports_the_music_gate():
    """`mic_gate_reason` が音楽でも理由を返す（タイマーと同じ口）。"""
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._timer_tool = None
    a._music_state = MagicMock(playing=True)
    assert "音楽" in EmbodiedAgent.mic_gate_reason(a)


# ── 寿命 ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_music_stops_by_itself_after_thirty_minutes():
    from familiar_agent.loop.music_watch import check_music_expired

    io = MagicMock()
    io.stop = AsyncMock(return_value=True)
    state = MagicMock(playing=True, started_at=1000.0)
    said: list[str] = []
    await check_music_expired(
        io=io, bus=MagicMock(), state=state, now=1000.0 + MUSIC_MAX_SEC, say=said.append
    )
    io.stop.assert_awaited_once()
    assert said and "30 分" in said[0]
    assert state.playing is False


@pytest.mark.asyncio
async def test_music_keeps_playing_before_the_limit():
    from familiar_agent.loop.music_watch import check_music_expired

    io = MagicMock()
    io.stop = AsyncMock(return_value=True)
    state = MagicMock(playing=True, started_at=1000.0)
    said: list[str] = []
    await check_music_expired(
        io=io, bus=MagicMock(), state=state, now=1000.0 + MUSIC_MAX_SEC - 1, say=said.append
    )
    io.stop.assert_not_awaited()
    assert said == []


# ── 減音 ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_music_ducks_while_paju_speaks_and_comes_back():
    from familiar_agent.loop.music_watch import duck_while_speaking

    io = MagicMock()
    io.status = AsyncMock(return_value={"playing": True, "volume": 0.8})
    io.set_volume = AsyncMock(return_value=True)
    state = MagicMock(playing=True, base_volume=None)

    async def speak():
        return "話した"

    got = await duck_while_speaking(
        io=io, bus=MagicMock(), state=state, speak=speak, sleep=AsyncMock()
    )
    assert got == "話した"
    # 絞ってから戻す（基準は人が変えた値）
    assert io.set_volume.await_args_list[0].args[1] == pytest.approx(0.8 * DUCK_RATIO)
    assert io.set_volume.await_args_list[-1].args[1] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_nothing_happens_when_no_music_plays():
    from familiar_agent.loop.music_watch import duck_while_speaking

    io = MagicMock()
    io.status = AsyncMock(return_value={})
    io.set_volume = AsyncMock()
    state = MagicMock(playing=False, base_volume=None)

    async def speak():
        return "話した"

    assert (
        await duck_while_speaking(
            io=io, bus=MagicMock(), state=state, speak=speak, sleep=AsyncMock()
        )
        == "話した"
    )
    io.set_volume.assert_not_awaited()


# ── 集音の門を通る言葉（音楽のときは音楽の言葉だけ）──────────────────────


@pytest.mark.asyncio
async def test_the_session_passes_music_words_when_music_is_the_reason():
    """門の理由が音楽なら、通すのは音楽の言葉（タイマーの操作語ではない）。"""
    from familiar_agent.realtime_stt_session import RealtimeSttSession

    session = MagicMock(spec=RealtimeSttSession)
    session.music_table = lambda: (("ケイマン", "spotify:playlist:aaa", False),)
    assert RealtimeSttSession._passes_gate(session, "音楽が鳴っている", "止めて")
    assert RealtimeSttSession._passes_gate(session, "音楽が鳴っている", "ケイマンかけて")
    assert not RealtimeSttSession._passes_gate(session, "音楽が鳴っている", "3 分測って")
    # タイマーのときは従来どおり（操作の言葉）
    assert RealtimeSttSession._passes_gate(session, "タイマー中", "止めて")
    assert not RealtimeSttSession._passes_gate(session, "タイマー中", "ケイマンかけて")


# ── 道具が主LLM へ渡る ─────────────────────────────────────────────────────


def test_the_music_tools_reach_the_main_llm():
    from familiar_agent.loop.event_loop import _FULL_ACTIONS, InformationProcessing
    from tests.test_event_loop import _agent as _base_agent, _turn

    for name in ("play_music", "stop_music", "next_track", "music_volume"):
        assert name in _FULL_ACTIONS, name
    a = _base_agent(stream_returns=[_turn([])])
    from familiar_agent.tools.music import MusicTool

    a._music_tool = MusicTool(io=MagicMock(), bus=lambda: MagicMock(), table=lambda: ())
    ip = InformationProcessing(a)
    names = {d.get("name") for d in ip._tools(actions=_FULL_ACTIONS, cache_tools=False)}
    assert {"play_music", "stop_music", "next_track", "music_volume"} <= names


def test_no_music_tools_without_the_device():
    from familiar_agent.loop.event_loop import _FULL_ACTIONS, InformationProcessing
    from tests.test_event_loop import _agent as _base_agent, _turn

    a = _base_agent(stream_returns=[_turn([])])
    a._music_tool = None
    ip = InformationProcessing(a)
    names = {d.get("name") for d in ip._tools(actions=_FULL_ACTIONS, cache_tools=False)}
    assert "play_music" not in names


# ── 声を出すあいだは音楽を絞る（DIF の結線）────────────────────────────────


@pytest.mark.asyncio
async def test_speaking_ducks_the_music_through_dif():
    """パジュが声を出すたびに絞って戻す（機械の反射・主LLM は通らない）。"""
    from familiar_agent.io.dif import DIF

    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: ok", None))
    ducked: list[str] = []

    async def ducker(speak):
        ducked.append("絞った")
        try:
            return await speak()
        finally:
            ducked.append("戻した")

    dif = DIF(tts=tts)
    dif.set_music_ducker(ducker)
    await dif.speak("こんにちは")
    assert ducked == ["絞った", "戻した"]
    tts.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_speaking_works_without_music():
    from familiar_agent.io.dif import DIF

    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: ok", None))
    dif = DIF(tts=tts)
    await dif.speak("こんにちは")
    tts.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_voice_still_comes_out_when_the_music_port_is_broken():
    """絞れなくても声は出す。**音楽は添え物で、声が本体**（2026-09-21・テストが捕まえた）。"""
    from familiar_agent.loop.music_watch import duck_while_speaking

    io = MagicMock()
    io.status = AsyncMock(side_effect=RuntimeError("D-Bus が無い"))
    io.set_volume = AsyncMock()
    state = MagicMock(playing=True)

    async def speak():
        return "話した"

    assert await duck_while_speaking(io=io, bus=MagicMock(), state=state, speak=speak) == "話した"
