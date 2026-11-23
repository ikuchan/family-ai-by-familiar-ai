"""マイクは PipeWire の入力の口を `pw-record` から読む（出-du 段 2・2026-11-23・本人の決定イ）。

11/23 9:37 の再起動から、パジュがマイクを直接（`hw:1,0`）掴み、PipeWire が同じ機器の入力の口を開こうとして「ビジー」で
ぶつかり、声を鳴らすたびに声と耳が止まった。YAMAHA を掴むのを PipeWire だけにし、読み取りは別のプロセスにする
（声の出口の `pw-play` と同じ形・固まっても止めるのは子だけ）。PipeWire の入力の口が無ければ、いまの道（sounddevice）。
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from familiar_agent.tools import mic

_SOURCE = "alsa_input.usb-Yamaha_Corporation_Yamaha_YVC-300-00.pro-input-0"


class _Out:
    """偽の標準出力。本物と同じく、止められたら終わる（裏の糸を試験のあとに残さない）。"""

    def __init__(self, data: bytes, terminated: threading.Event):
        self._buf = io.BytesIO(data)
        self._terminated = terminated

    def read(self, n):
        return b"" if self._terminated.is_set() else self._buf.read(n)


class _FakeRecord:
    """偽の `pw-record`。48 kHz の音を標準出力に出す。止められたら終わる。"""

    def __init__(self, args, **kw):
        self.args = args
        self.terminated = threading.Event()
        self.stdout = _Out(
            (np.ones(48000, dtype=np.int16) * 1000).tobytes(), self.terminated
        )  # 1 秒
        self.returncode = None

    def terminate(self):
        self.terminated.set()
        self.returncode = -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        if self.returncode is None and timeout is not None:
            raise subprocess.TimeoutExpired("pw-record", timeout)
        return self.returncode

    def poll(self):
        return self.returncode


def _capture(monkeypatch, *, source=_SOURCE):
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "YVC")
    monkeypatch.setattr(mic, "_pw_record_binary", lambda: "/usr/bin/pw-record")
    monkeypatch.setattr(mic, "_pw_source_name", lambda wanted: source)
    made: list = []

    def _popen(args, **kw):
        p = _FakeRecord(args, **kw)
        made.append(p)
        return p

    monkeypatch.setattr(mic.subprocess, "Popen", _popen)
    return made


def test_the_mic_is_read_from_pw_record(monkeypatch):
    made = _capture(monkeypatch)
    got: list[bytes] = []

    async def on_audio(b):
        got.append(b)

    async def go():
        cap = mic.MicCapture(on_audio=on_audio)
        cap.start(asyncio.get_running_loop())
        for _ in range(50):
            if sum(len(b) for b in got) >= 16000 * 2 * 0.9:
                break
            await asyncio.sleep(0.02)
        cap.stop()
        return cap

    cap = asyncio.run(go())
    args = made[0].args
    assert args[0] == "/usr/bin/pw-record" and f"--target={_SOURCE}" in args
    assert "--raw" in args and "--rate=48000" in args and "--channels=1" in args and args[-1] == "-"
    total = sum(len(b) for b in got)
    assert 16000 * 2 * 0.9 <= total <= 16000 * 2 * 1.1  # 48 kHz の 1 秒が 16 kHz の 1 秒になった
    assert cap.device_name == _SOURCE  # 機器の見張り（名前に YVC を含むか）がそのまま通る


def test_stop_ends_the_child_and_returns_at_once(monkeypatch):
    made = _capture(monkeypatch)

    async def go():
        cap = mic.MicCapture(on_audio=lambda b: asyncio.sleep(0))
        cap.start(asyncio.get_running_loop())
        started = time.monotonic()
        cap.stop()
        return time.monotonic() - started

    took = asyncio.run(go())
    assert made[0].terminated.is_set()
    assert took < 2.0


def test_without_a_pipewire_input_the_old_path_is_used(monkeypatch):
    """**反証の側**：PipeWire の入力の口が無ければ、`pw-record` は起こさず sounddevice の道で開く。"""
    made = _capture(monkeypatch, source=None)
    sd = MagicMock()
    sd.query_devices.side_effect = lambda *a, **k: (
        [{"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "max_input_channels": 1, "hostapi": 0}]
        if not k
        else {"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "default_samplerate": 48000}
    )
    sd.query_hostapis.return_value = [{"name": "ALSA"}]
    # sounddevice の 1 つだけを差し替える（`patch.dict` は抜けるときに、試験中に読み込んだ soxr まで消し、次の試験で
    # C の拡張を読み直して worker ごと落ちた）。
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    loop = asyncio.new_event_loop()
    with patch.object(mic, "probe_sounddevice_input", return_value=(True, "ok")):
        mic.MicCapture(on_audio=lambda b: asyncio.sleep(0)).start(loop)
    loop.close()
    assert made == []
    assert sd.RawInputStream.called


def test_a_recorder_that_ends_at_once_falls_back_to_the_old_path(monkeypatch, caplog):
    """起こしてすぐ終わった（行き先が無い・PipeWire に届かない）なら、警告を残して直接の道（sounddevice）で開く。

    始まりかけで耳が聞こえないまま残らない（見直し 2 回目）。直接の道は取り合いで声が止まることがあるが、出-du の見張りで抜ける。
    """
    made = _capture(monkeypatch)

    class _Dead(_FakeRecord):
        def __init__(self, args, **kw):
            super().__init__(args, **kw)
            self.returncode = 1

    monkeypatch.setattr(
        mic.subprocess, "Popen", lambda args, **kw: made.append(_Dead(args)) or made[-1]
    )
    sd = MagicMock()
    sd.query_devices.side_effect = lambda *a, **k: (
        [{"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "max_input_channels": 1, "hostapi": 0}]
        if not k
        else {"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "default_samplerate": 48000}
    )
    sd.query_hostapis.return_value = [{"name": "ALSA"}]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    loop = asyncio.new_event_loop()
    with (
        caplog.at_level("WARNING"),
        patch.object(mic, "probe_sounddevice_input", return_value=(True, "ok")),
    ):
        mic.MicCapture(on_audio=lambda b: asyncio.sleep(0)).start(loop)
    loop.close()
    assert len(made) == 1  # pw-record は一度だけ試した
    assert sd.RawInputStream.called
    assert "pw-record で始められなかった" in caplog.text


def test_the_recorder_never_falls_back_to_another_source(monkeypatch):
    """行き先が消えても、既定の入力へ付け替えない印を渡す（付け替わると見張りが気づけない）。"""
    made = _capture(monkeypatch)

    async def go():
        cap = mic.MicCapture(on_audio=lambda b: asyncio.sleep(0))
        cap.start(asyncio.get_running_loop())
        cap.stop()

    asyncio.run(go())
    props = made[0].args[made[0].args.index("-P") + 1]
    assert "node.dont-fallback = true" in props and "node.dont-reconnect = true" in props
