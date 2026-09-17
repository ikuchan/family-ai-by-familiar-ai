"""タイマーの音 `src/familiar_agent/sounds/timer_alarm.wav` を合成する（知-n-ろ・2026-09-17）。

拾ってきた音源は許諾の確認と出典の記録が要るので、自分で作る。キッチンタイマーふうの
「ピピピッ」（1 kHz と 1.3 kHz の短音 3 連＋休み）を 1 周期 1.0 秒で作り、鳴らす側
（`io/dif.py` の `ring_timer`）がこれを `TIMER_RING_SEC` のあいだ繰り返す。

使い方：`uv run python scripts/gen_timer_alarm.py`
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

OUT = (
    Path(__file__).resolve().parent.parent / "src" / "familiar_agent" / "sounds" / "timer_alarm.wav"
)
RATE = 48000  # YVC-300 などが 48 kHz を求める。再生側は機器の rate へ合わせ直すが、元も揃えておく
BEEP_SEC = 0.09
GAP_SEC = 0.06
PERIOD_SEC = 1.0
FREQS = (1000.0, 1300.0)  # 2 音を重ねて「電子音」らしくする
AMPLITUDE = 0.6  # 飽和させない（`TIMER_VOICE_GAIN` で上げる余地を残す）


def _beep(sec: float) -> np.ndarray:
    t = np.arange(int(RATE * sec)) / RATE
    wave_ = sum(np.sin(2 * np.pi * f * t) for f in FREQS) / len(FREQS)
    # 頭と尻を 5 ms で絞り、クリック音を出さない
    ramp = int(RATE * 0.005)
    env = np.ones_like(wave_)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return wave_ * env * AMPLITUDE


def build_period() -> np.ndarray:
    """1 周期ぶん（ピピピッ ＋ 休み）を float32 で返す。"""
    parts: list[np.ndarray] = []
    for i in range(3):
        parts.append(_beep(BEEP_SEC))
        if i < 2:
            parts.append(np.zeros(int(RATE * GAP_SEC)))
    body = np.concatenate(parts)
    rest = np.zeros(max(0, int(RATE * PERIOD_SEC) - len(body)))
    return np.concatenate([body, rest]).astype(np.float32)


def write(path: Path = OUT) -> Path:
    data = build_period()
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    p = write()
    print(f"{p}（{p.stat().st_size} bytes・{RATE} Hz・{PERIOD_SEC} 秒）")
