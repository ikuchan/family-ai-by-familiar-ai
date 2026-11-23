"""音の機器の名前と選び方（2026-11-23）。マイクも声の出口も、ここを通る。

名前は `config.AgentConfig.audio_device`（家の YAMAHA YVC-300）。以前は `.env` の `AUDIO_INPUT_DEVICE`・
`AUDIO_OUTPUT_DEVICE` を 7 か所で読んでいた（`.env` は設定ファイルではない）。

選び方は、名前が合う機器のうち PipeWire（PulseAudio の口）を通るものを先にし、無ければ直接の機器（ALSA の `hw:`）。
直接・排他で掴むと、PipeWire が同じ機器を開くときにぶつかる（声の出口は知-ak-ろ・2026-10-07、マイクは出-du・
2026-11-23）。マイクでは、スピーカーの写し（名前が `.monitor` で終わる）を選ばない（パジュ自身の声しか入らない）。
"""

from __future__ import annotations

from typing import Literal


def device_name() -> str:
    """音の機器の名前（名前の一部。機器の一覧の名前に含まれるかで選ぶ）。"""
    from ..config import AgentConfig

    return AgentConfig.audio_device


def pick_device(
    devices: "list[dict]", apis: "list[dict]", wanted: str, *, kind: Literal["input", "output"]
) -> "int | None":
    """名前に `wanted` を含む `kind` の機器の番号。PipeWire を通るものを先に選ぶ。無ければ None。"""
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    hits = [
        i
        for i, d in enumerate(devices)
        if wanted.lower() in str(d["name"]).lower()
        and d[key] > 0
        and not (kind == "input" and str(d["name"]).endswith(".monitor"))
    ]
    routed = [i for i in hits if is_pipewire(apis, devices[i])]
    chosen = routed or hits
    return chosen[0] if chosen else None


def is_pipewire(apis: "list[dict]", device: dict) -> bool:
    """その機器が PipeWire（PulseAudio の口）を通るか。"""
    return "pulse" in str(apis[device["hostapi"]]["name"]).lower()
