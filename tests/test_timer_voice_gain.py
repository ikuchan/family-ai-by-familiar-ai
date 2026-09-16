"""タイマーが鳴ったときだけ声を大きくする（`TIMER_VOICE_GAIN`・2026-09-16）。

機器の音量には触らず、**その 1 回の再生データにだけ**倍率を掛ける（マイクの入力ゲインと同じ
仕組み・範囲を超えた分は飽和）。鳴り終わった次の発話は元の音量で、戻す処理は要らない。
倍率は `.env` の `TIMER_VOICE_GAIN`（既定 1.0＝いまと同じ・この機体は 1.5）。

経路：タイマー起点の求め → `_speak` → `DIF.speak(text, gain=)` → `TTSTool.say(gain=)` →
sounddevice の再生で掛ける。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from familiar_agent.config import AgentConfig
from familiar_agent.io.dif import DIF
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.tools import tts as tts_mod


def test_the_gain_is_read_from_env_and_defaults_to_one():
    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().timer_voice_gain == 1.0
    with patch.dict(os.environ, {"TIMER_VOICE_GAIN": "1.5"}, clear=True):
        assert AgentConfig().timer_voice_gain == 1.5


def test_playback_scales_the_samples_and_saturates(tmp_path):
    import soundfile as sf

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.array([0.2, -0.2, 0.9], dtype="float32"), 16000)
    played: list = []
    sd = MagicMock()
    sd.query_devices = MagicMock(return_value={"default_samplerate": 16000})
    sd.play = lambda data, rate, device=None: played.append(np.asarray(data))
    sd.wait = lambda: None
    with (
        patch.dict("sys.modules", {"sounddevice": sd}),
        patch.object(tts_mod, "_resolve_output_device", return_value=None),
    ):
        ok = asyncio.run(tts_mod._play_via_sounddevice(str(wav), gain=1.5))
    assert ok and len(played) == 1
    got = played[0]
    assert abs(got[0] - 0.3) < 1e-3 and abs(got[1] + 0.3) < 1e-3
    assert abs(got[2] - 1.0) < 1e-6  # 0.9×1.5 は飽和


def test_say_passes_the_gain_to_playback():
    tool = tts_mod.TTSTool("k", "v", engine="elevenlabs", output="local")
    tool._play_local_gain_seen: list[float] = []
    resp = MagicMock()
    resp.status = 200
    resp.headers = {"Content-Type": "audio/pcm"}
    resp.read = AsyncMock(return_value=b"\x00\x00" * 1600)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    seen: list[float] = []

    async def fake_play_local(path, gain=1.0):
        seen.append(gain)
        return True

    with (
        patch("aiohttp.ClientSession", return_value=session),
        patch.object(tts_mod, "_play_local", new=fake_play_local),
    ):
        asyncio.run(tool.call("say", {"text": "時間だよ", "gain": 1.5}))
        asyncio.run(tool.call("say", {"text": "こんにちは"}))
    assert seen == [1.5, 1.0]


def test_dif_forwards_the_gain_only_when_it_matters():
    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: x", None))
    dif = DIF(tts=tts, search=None, fetch=None, mcp=None, ip=None)
    asyncio.run(dif.speak("時間だよ", gain=1.5))
    asyncio.run(dif.speak("こんにちは"))
    calls = [c.args for c in tts.call.await_args_list]
    assert calls[0] == ("say", {"text": "時間だよ", "gain": 1.5})
    assert calls[1] == ("say", {"text": "こんにちは"})


def test_only_a_timer_request_speaks_louder():
    a = MagicMock()
    a.config.timer_voice_gain = 1.5
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "機器"
    ip._req.request_text = "[タイマー] タイマー：「パパのタイマー」の時間"
    assert ip._voice_gain() == 1.5
    ip._req.trigger_kind = "発話"
    ip._req.request_text = "おはよう"
    assert ip._voice_gain() == 1.0
    ip._req.trigger_kind = "機器"
    ip._req.request_text = "[入室] パパ が来た"
    assert ip._voice_gain() == 1.0
