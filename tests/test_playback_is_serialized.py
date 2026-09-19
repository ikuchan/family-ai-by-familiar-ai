"""音声の再生は 1 本ずつ（出-ad・2026-09-19）。

タイマーが鳴った瞬間、音（`DIF.ring`・1 秒の wav）と知らせの声（`DIF.speak`・mp3）が同じ機器 `hw:1,0`（排他）を
別スレッドから同時に開き、PortAudio（ALSA）が `Device unavailable [-9985]` のあと `double free or corruption (out)` で
プロセスごと落ちた（実機 11:39・quiet の回は声が silent だったので重ならなかった）。`_play_via_sounddevice` は
プロセス内の鍵（`_PLAYBACK_LOCK`）で直列にする。音の 1 秒 → 声 → 音の次の 1 秒、と交互になる。
"""

from __future__ import annotations

import asyncio
import threading
import time

from familiar_agent.tools import tts as tts_mod


def test_two_plays_never_overlap(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    import numpy as np
    import soundfile as sf

    sf.write(str(wav), np.zeros(4800, dtype="float32"), 48000)
    state = {"open": 0, "overlap": 0}
    lock = threading.Lock()

    class _SD:
        default = type("d", (), {"device": (0, 0)})()

        @staticmethod
        def query_devices(idx=None, kind=None):
            return {"default_samplerate": 48000.0, "name": "fake", "max_output_channels": 1}

        @staticmethod
        def play(data, rate, device=None):
            with lock:
                if state["open"]:
                    state["overlap"] += 1
                state["open"] += 1

        @staticmethod
        def wait():
            time.sleep(0.05)
            with lock:
                state["open"] -= 1

    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", _SD)
    monkeypatch.setattr(tts_mod, "_resolve_output_device", lambda: None)

    async def both():
        return await asyncio.gather(
            *[tts_mod._play_via_sounddevice(str(wav), 1.0) for _ in range(6)]
        )

    results = asyncio.run(both())
    assert all(results) and state["overlap"] == 0  # 6 本が重ならずに順に鳴った
