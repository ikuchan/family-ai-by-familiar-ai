"""音の機器の名前は `config.py` に 1 つ、選び方は `tools/audio_devices.py` に 1 つ（2026-11-23・本人「4 はやって」）。

名前は `.env` の `AUDIO_INPUT_DEVICE`・`AUDIO_OUTPUT_DEVICE` を 7 か所で読み、「PipeWire の口を先に選ぶ」規則はマイクと声の
出口で 3 か所に写されていた（`.env` は設定ファイルではない、に反していた）。
"""

from __future__ import annotations

from familiar_agent.config import AgentConfig
from familiar_agent.tools import audio_devices

_APIS = [{"name": "ALSA"}, {"name": "PulseAudio"}]
_HW = {
    "name": "Yamaha YVC-300: USB Audio (hw:1,0)",
    "max_input_channels": 1,
    "max_output_channels": 2,
    "hostapi": 0,
}
_MON = {
    "name": "alsa_output.usb-Yamaha_YVC-300-00.pro-output-0.monitor",
    "max_input_channels": 1,
    "max_output_channels": 0,
    "hostapi": 1,
}
_IN = {
    "name": "alsa_input.usb-Yamaha_YVC-300-00.pro-input-0",
    "max_input_channels": 1,
    "max_output_channels": 0,
    "hostapi": 1,
}
_OUT = {
    "name": "alsa_output.usb-Yamaha_YVC-300-00.pro-output-0",
    "max_input_channels": 0,
    "max_output_channels": 2,
    "hostapi": 1,
}


def test_the_name_comes_from_config_not_env(monkeypatch):
    import dataclasses

    default = next(f.default for f in dataclasses.fields(AgentConfig) if f.name == "audio_device")
    assert default == "YVC-300"
    # 試験の共通の差し替え（conftest）で空にしてあるので、既定に戻してから見る。
    monkeypatch.setattr(AgentConfig, "audio_device", default)
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "Other")
    monkeypatch.setenv("AUDIO_OUTPUT_DEVICE", "Other")
    assert audio_devices.device_name() == "YVC-300"


def test_the_input_prefers_pipewire_and_skips_the_monitor():
    assert audio_devices.pick_device([_HW, _MON, _IN, _OUT], _APIS, "YVC-300", kind="input") == 2


def test_the_output_prefers_pipewire():
    assert audio_devices.pick_device([_HW, _MON, _IN, _OUT], _APIS, "YVC-300", kind="output") == 3


def test_without_pipewire_the_direct_device_is_used():
    """**反証の側**：PipeWire の口が無ければ、直接の機器を使う。"""
    assert audio_devices.pick_device([_HW], _APIS, "YVC-300", kind="input") == 0
    assert audio_devices.pick_device([_HW], _APIS, "YVC-300", kind="output") == 0
    assert audio_devices.pick_device([_HW], _APIS, "nothing", kind="output") is None
