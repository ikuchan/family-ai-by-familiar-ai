"""声の出口に PipeWire 側の Yamaha を選ぶ（知-ak-ろ 段 2・2026-10-07 実機）。

声は名前に `AUDIO_OUTPUT_DEVICE`（`YVC-300`）を含む最初の機器を選び、ALSA の `hw:1,0` を直に・排他で開いていた。
音楽（spotifyd）が PipeWire で Yamaha を握っているあいだ、声は `Device unavailable [PaErrorCode -9985]` で 10 回消えた。
PulseAudio（PipeWire）側にも同じ Yamaha の出口があれば、そちらを選ぶ（音楽と声を PipeWire が混ぜる）。無ければいまどおり。
"""

from __future__ import annotations

import sys
import types

from familiar_agent.tools import tts

_APIS = [{"name": "ALSA"}, {"name": "OSS"}, {"name": "PulseAudio"}]
_ALSA = {"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "hostapi": 0, "max_output_channels": 1}
_PULSE = {
    "name": "alsa_output.usb-Yamaha_Corporation_Yamaha_YVC-300-00.mono-fallback",
    "hostapi": 2,
    "max_output_channels": 1,
}
_OTHER = {"name": "pipewire", "hostapi": 0, "max_output_channels": 128}


def _sd(monkeypatch, devices):
    fake = types.SimpleNamespace(query_devices=lambda: devices, query_hostapis=lambda: _APIS)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_the_pipewire_yamaha_is_chosen_over_the_raw_device(monkeypatch):
    monkeypatch.setenv("AUDIO_OUTPUT_DEVICE", "YVC-300")
    _sd(monkeypatch, [_OTHER, _ALSA, _PULSE])
    assert tts._resolve_output_device() == 2


def test_without_pipewire_the_raw_device_is_used(monkeypatch):
    monkeypatch.setenv("AUDIO_OUTPUT_DEVICE", "YVC-300")
    _sd(monkeypatch, [_OTHER, _ALSA])
    assert tts._resolve_output_device() == 1


def test_without_a_name_the_default_is_used(monkeypatch):
    monkeypatch.delenv("AUDIO_OUTPUT_DEVICE", raising=False)
    monkeypatch.delenv("AUDIO_INPUT_DEVICE", raising=False)
    _sd(monkeypatch, [_ALSA, _PULSE])
    assert tts._resolve_output_device() is None
