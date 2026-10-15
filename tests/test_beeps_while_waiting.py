"""待ちのあいだの電子音（出-bu・2026-10-12・本人の決定：間隔 0.5 秒→1 秒）。音楽中も鳴らす（出-cl・2026-10-15・本人の決定カ）。

10/12 22:15「明日の天気は?」では、合図「はい」から最初のつなぎまで 13 秒無音だった（調べもの約 5 秒・調停の聞き直し
約 2 秒・主LLM 約 5 秒）。機械音 A は最初の判断のあいだ（0.5 秒ほど）しか鳴らない。合図が鳴り終わってから答えの声が
始まるまで、機械音 A を間隔をあけて 1 回ずつ鳴らす。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, interval=0.05):
    a = _agent(stream_returns=[])
    a.config.wait_beep_interval_seconds = interval
    a._music_state = MagicMock(playing=False)
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    beeps: list[float] = []

    async def beep():
        beeps.append(asyncio.get_running_loop().time())
        return True

    ip._dif.beep = beep  # type: ignore[method-assign]
    return ip, beeps


def _run(ip, *, seconds, during=None):
    async def go():
        ip._start_beeps()
        if during is not None:
            await during()
        await asyncio.sleep(seconds)
        ip._stop_beeps()
        await ip.close()

    asyncio.run(go())


def test_it_beeps_while_waiting():
    ip, beeps = _ip()
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3


def test_beeps_while_music_plays():
    """つなぎの声をやめたので、音楽中に鳴らさないと待ちのあいだ何も鳴らない（出-cl・本人の決定カ）。音楽は絞らない。"""
    ip, beeps = _ip()
    ip._agent._music_state.playing = True
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3


def test_beeps_rest_while_a_cue_sounds():
    """**反証の側**：合図の声が鳴っているあいだは休む（つなぎの声は出-cl 段 3 で撤去）。"""
    ip, beeps = _ip()

    async def voice():
        ip._cue_sound = asyncio.ensure_future(asyncio.sleep(10))

    _run(ip, seconds=0.3, during=voice)
    assert beeps == []


def test_the_answer_stops_the_beeps(caplog):
    ip, beeps = _ip()
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    after: list[int] = []

    async def go():
        ip._start_beeps()
        await asyncio.sleep(0.2)
        await ip._speak("明日は晴れです")
        after.append(len(beeps))
        await asyncio.sleep(0.2)
        after.append(len(beeps))
        await ip.close()

    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert after[0] == after[1] and after[0] >= 1
    assert "待ちの電子音" in caplog.text


def test_closing_the_request_stops_the_beeps():
    ip, beeps = _ip()
    counts: list[int] = []

    async def go():
        ip._start_beeps()
        await asyncio.sleep(0.15)
        ip._stop_wait_watch()  # 求めが閉じる・打ち切られる道はここを通る
        counts.append(len(beeps))
        await asyncio.sleep(0.15)
        counts.append(len(beeps))
        await ip.close()

    asyncio.run(go())
    assert counts[0] == counts[1]


def _decided(final: str) -> bool:
    from familiar_agent.loop.event_loop import _ReactionCue

    ip, _ = _ip()
    started: list[bool] = []
    ip._start_beeps = lambda: started.append(True)  # type: ignore[method-assign]

    async def go():
        ip._dif.play_cue = lambda cue: asyncio.get_running_loop().create_future()  # type: ignore[method-assign]
        _ReactionCue(ip, MagicMock()).decided(final)
        await ip.close()

    asyncio.run(go())
    return bool(started)


def test_a_cue_starts_the_beeps():
    assert _decided("search_deferred") is True


def test_silence_starts_no_beeps():
    """**反証の側**：黙ると決めたら（合図なし）、電子音も鳴らさない。"""
    assert _decided("silent") is False


def test_a_quiet_play_logs_at_debug(tmp_path, monkeypatch, caplog):
    import numpy as np
    import soundfile as sf

    from familiar_agent.tools import tts

    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: None)

    class _Proc:
        returncode = 0

        def __init__(self, *a, **k):
            pass

        def communicate(self, timeout=None):
            import time

            time.sleep(0.12)
            return b"", b""

        def kill(self):
            pass

    monkeypatch.setattr(tts.subprocess, "Popen", _Proc)
    p = tmp_path / "b.wav"
    sf.write(str(p), np.zeros(int(0.1 * 24000), dtype="float32"), 24000)

    with caplog.at_level("DEBUG"):
        asyncio.run(tts._play_via_sounddevice(str(p), 1.0, quiet=True))
    lines = [r for r in caplog.records if "声を鳴らした" in r.getMessage()]
    assert lines and all(r.levelname == "DEBUG" for r in lines)


def test_the_interval_comes_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        interval = AgentConfig().wait_beep_interval_seconds  # 設定を丸ごと表示させない（鍵が出る）
    assert interval == 1.0  # 0.5 秒は短すぎた（本人 2026-10-13）
