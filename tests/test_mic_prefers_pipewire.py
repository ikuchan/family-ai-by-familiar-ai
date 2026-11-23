"""マイクは、名前が合う機器のうち PipeWire を通るものを先に選ぶ（出-du 段 2・2026-11-23）。

11/23 9:37 の再起動から PipeWire が YAMAHA に入力の口を持ち、声を鳴らすたびにその口がマイクを開こうとした。パジュが
同じマイクを直接（`hw:1,0`）掴んでいたので「ビジー」で失敗し、スピーカーの再生も始まらず、声と耳が止まった。
"""

from __future__ import annotations

from familiar_agent.tools.mic import pick_input_device

_APIS = [{"name": "ALSA"}, {"name": "PulseAudio"}, {"name": "JACK Audio Connection Kit"}]
_HW = {"name": "Yamaha YVC-300: USB Audio (hw:1,0)", "max_input_channels": 1, "hostapi": 0}
_PLUG = {"name": "sysdefault:CARD=YVC300", "max_input_channels": 1, "hostapi": 0}
_JACK = {"name": "Yamaha YVC-300 Pro", "max_input_channels": 1, "hostapi": 2}
_MON = {
    "name": "alsa_output.usb-Yamaha_Corporation_Yamaha_YVC-300-00.pro-output-0.monitor",
    "max_input_channels": 1,
    "hostapi": 1,
}
_PW = {
    "name": "alsa_input.usb-Yamaha_Corporation_Yamaha_YVC-300-00.pro-input-0",
    "max_input_channels": 1,
    "hostapi": 1,
}
_OUT = {
    "name": "alsa_output.usb-Yamaha_Corporation_Yamaha_YVC-300-00.pro-output-0",
    "max_input_channels": 0,
    "hostapi": 1,
}


def test_the_pipewire_input_is_chosen_over_the_direct_device():
    assert pick_input_device([_OUT, _HW, _PW], _APIS, "YVC") == 2


def test_the_speaker_monitor_is_never_the_mic():
    """実機の並び（11/23）：スピーカーの写し（22）が本物のマイク（23）より先に来る。写しはパジュ自身の声しか入らない。"""
    assert pick_input_device([_OUT, _HW, _MON, _PW], _APIS, "YVC") == 3


def test_other_non_pulse_entries_do_not_count_as_pipewire():
    """名前に `(hw:` が無くても、PulseAudio の口でなければ PipeWire を通る機器とは見ない（JACK・ALSA の差し込み）。"""
    assert pick_input_device([_JACK, _PLUG, _HW, _PW], _APIS, "YVC") == 3


def test_without_a_pipewire_input_the_direct_device_is_used():
    """**反証の側**：PipeWire に入力の口が無い起動（11/14〜11/23 朝）では、いまどおり直接の機器を使う。"""
    assert pick_input_device([_OUT, _MON, _HW], _APIS, "YVC") == 2


def test_no_match_gives_none():
    assert pick_input_device([_OUT, _HW], _APIS, "nothing") is None
