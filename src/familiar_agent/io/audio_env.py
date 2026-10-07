"""音の口をこの機体にする（知-ak 段 4・知-ak-ろ・2026-10-07 実機）。

アプリを遠隔デスクトップ（Chrome Remote Desktop）のターミナルから起動すると、その接続用の音の口を指す環境変数
（`PULSE_RUNTIME_PATH`・`PULSE_SINK`・`PIPEWIRE_REMOTE`）を引き継ぐ。spotifyd の音は遠隔デスクトップへ流れ、パジュの声
（PortAudio）からは PipeWire 側の Yamaha の出口が見えず、直に開く `hw:1,0` を選んで、音楽が鳴っているあいだ
`Device unavailable` で消えた。3 つを外し、この機体の音の口（`<XDG_RUNTIME_DIR>/pulse/native`・既定の出口は Yamaha）を
`PULSE_SERVER` で指す。アプリ自身は起動の最初に（`apply_local_audio`）、spotifyd は立ち上げるときに（`local_audio_env`）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

#: 遠隔デスクトップの接続用の音の口を指す環境変数。
REMOTE_AUDIO_VARS = ("PULSE_RUNTIME_PATH", "PULSE_SINK", "PIPEWIRE_REMOTE")


def _local_socket(env: "Mapping[str, str]") -> Path:
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "pulse" / "native"


def local_audio_env(env: "Mapping[str, str]") -> "dict[str, str]":
    """`env` の写しから遠隔デスクトップの音の口を外し、この機体の音の口を指す（無ければ外すだけ）。"""
    out = {k: v for k, v in env.items() if k not in REMOTE_AUDIO_VARS}
    native = _local_socket(env)
    if native.exists():
        out["PULSE_SERVER"] = f"unix:{native}"
    else:
        logger.warning("この機体の音の口（%s）が見つからない。既定の口で鳴らす", native)
    return out


def apply_local_audio() -> bool:
    """プロセスの環境を、この機体の音の口へ向ける。遠隔デスクトップの環境が無ければ何もしない（False）。

    PortAudio は読み込んだときの環境で音の口を決めるので、sounddevice を読む前（起動の最初）に呼ぶ。
    """
    if not any(k in os.environ for k in REMOTE_AUDIO_VARS):
        return False
    new = local_audio_env(os.environ)
    for k in REMOTE_AUDIO_VARS:
        os.environ.pop(k, None)
    if "PULSE_SERVER" in new:
        os.environ["PULSE_SERVER"] = new["PULSE_SERVER"]
    logger.info(
        "音の口：遠隔デスクトップの口を外し、この機体の口へ向けた（%s）",
        new.get("PULSE_SERVER", "既定"),
    )
    return True
