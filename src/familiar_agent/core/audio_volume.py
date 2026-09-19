"""出力機器のミキサーの音量を読む（環-q・2026-09-19）。

YVC-300 の ALSA ミキサー `PCM` が 40%（-30 dB）で、タイマーの音も声もほぼ無音だった（実機 21:55・22:13）。
起動時に読んで低ければ知らせる。**値は変えない**（勝手に上げると夜に驚かせる）。読むのは `amixer`（ALSA の
道具）で、機器の card 番号は `aplay -l` から名前で引く。無ければ None（黙る）。
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

_PERCENT = re.compile(r"\[(\d{1,3})%\]")
_CARD = re.compile(r"^(?:card|カード) (\d+): (\S+) \[([^\]]*)\]", re.M)


def parse_amixer(text: str) -> "int | None":
    """`amixer sget PCM` の出力から音量（%）を読む。無ければ None。"""
    m = _PERCENT.search(text or "")
    return int(m.group(1)) if m else None


def _card_of(name: str) -> "int | None":
    if not shutil.which("aplay"):
        return None
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    key = name.lower().replace("-", "")
    for m in _CARD.finditer(out):
        if key in (m.group(2) + m.group(3)).lower().replace("-", ""):
            return int(m.group(1))
    return None


def read_percent(device_name: str, control: str = "PCM") -> "int | None":
    """機器（名前の一部・例 "YVC-300"）のミキサーの音量（%）。機器か道具が無ければ None。"""
    if not device_name or not shutil.which("amixer"):
        return None
    card = _card_of(device_name)
    if card is None:
        return None
    try:
        out = subprocess.run(
            ["amixer", "-c", str(card), "sget", control], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    return parse_amixer(out)
