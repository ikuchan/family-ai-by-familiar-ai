"""声を `pw-play` で鳴らし、鳴り終わりの待ちに上限を付ける（知-ak-ろ 段 4・2026-10-07 実機）。

段 2 で声を PortAudio の PulseAudio の口から出したところ、20:34:33 の「少々お待ちください。」の鳴り終わりの待ち
（`sd.wait()`）が戻らなかった。減音が戻らず（音楽は 0.12 のまま）、続く返事も出ず、GUI を閉じてもプロセスが終わら
なかった。PipeWire 付属の `pw-play` を別のプロセスで呼び（固まっても止めるのは子だけ）、声の長さ＋余裕（5 秒〔仮〕・
`TTS_PLAY_MARGIN_SEC`）で打ち切る。`pw-play` が無ければ PortAudio の道で鳴らし、そちらにも同じ上限を付ける。
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from familiar_agent.tools import tts


def _wav(tmp_path, seconds=1.0, rate=24000):
    p = tmp_path / "v.wav"
    sf.write(str(p), np.zeros(int(seconds * rate), dtype="float32"), rate)
    return p


class _Proc:
    def __init__(self, args, finishes=True):
        self.args = args
        self.finishes = finishes
        self.killed = False

    def wait(self, timeout=None):
        if self.finishes:
            return 0
        import subprocess

        raise subprocess.TimeoutExpired(self.args, timeout)

    def kill(self):
        self.killed = True


@pytest.fixture
def pw(monkeypatch):
    seen: dict = {"procs": [], "timeouts": []}
    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: "alsa_output.usb-Yamaha_YVC-300.mono-fallback")

    def popen(args, **kw):
        p = _Proc(args, finishes=seen.get("finishes", True))
        orig = p.wait

        def wait(timeout=None):
            seen["timeouts"].append(timeout)
            return orig(timeout)

        p.wait = wait  # type: ignore[method-assign]
        seen["procs"].append(p)
        return p

    monkeypatch.setattr(tts.subprocess, "Popen", popen)
    return seen


def test_a_wav_is_played_with_pw_play(pw, tmp_path, monkeypatch):
    monkeypatch.delenv("TTS_PLAY_MARGIN_SEC", raising=False)
    wav = _wav(tmp_path, seconds=2.0)
    assert asyncio.run(tts._play_via_sounddevice(str(wav), gain=1.5)) is True
    (p,) = pw["procs"]
    args = p.args
    assert args[0] == "/usr/bin/pw-play"
    assert "--target=alsa_output.usb-Yamaha_YVC-300.mono-fallback" in args
    assert "--volume=1.5" in args and "--media-role=Communication" in args
    assert pw["timeouts"] == [pytest.approx(7.0, abs=0.05)]  # 2 秒＋5 秒


def test_without_a_target_it_uses_the_default_output(pw, tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_pw_target", lambda: None)
    asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path)), gain=1.0))
    assert not any(a.startswith("--target") for a in pw["procs"][0].args)


def test_a_stuck_play_is_cut_at_the_limit(pw, tmp_path, caplog):
    pw["finishes"] = False
    with caplog.at_level("WARNING"):
        assert asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path)), gain=1.0)) is False
    assert pw["procs"][0].killed
    assert "打ち切った" in caplog.text


def test_an_mp3_is_turned_into_a_temp_wav_and_removed(pw, tmp_path, monkeypatch):
    (tmp_path / "x").mkdir()
    made = str(_wav(tmp_path / "x", 1.0))
    monkeypatch.setattr(tts, "_mp3_to_wav", lambda path: made)
    assert asyncio.run(tts._play_via_sounddevice(str(tmp_path / "a.mp3"), gain=1.0)) is True
    assert pw["procs"][0].args[-1] == made
    import os

    assert not os.path.exists(made)


def test_without_pw_play_portaudio_is_used_with_a_limit(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tts, "_pw_play_binary", lambda: None)
    monkeypatch.setenv("TTS_PLAY_MARGIN_SEC", "0.2")
    stopped = threading.Event()

    class _SD:
        @staticmethod
        def query_devices(idx=None, kind=None):
            return {"default_samplerate": 24000.0, "name": "fake", "max_output_channels": 1}

        @staticmethod
        def play(data, rate, device=None):
            pass

        @staticmethod
        def wait():
            stopped.wait(5)  # 固まった待ち（stop されるまで戻らない）

        @staticmethod
        def stop():
            stopped.set()

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", _SD)
    monkeypatch.setattr(tts, "_resolve_output_device", lambda: None)
    t0 = time.monotonic()
    with caplog.at_level("WARNING"):
        ok = asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path, seconds=0.1)), gain=1.0))
    assert ok is False and time.monotonic() - t0 < 2.0
    assert stopped.is_set() and "打ち切った" in caplog.text


def test_the_margin_setting(monkeypatch):
    from familiar_agent.config import TTSConfig

    monkeypatch.delenv("TTS_PLAY_MARGIN_SEC", raising=False)
    assert TTSConfig().play_margin_sec == 5.0
