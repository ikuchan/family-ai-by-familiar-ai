#!/usr/bin/env python
"""話しかけたときの合図の音 `src/familiar_agent/sounds/react_*.wav` を作る（出-bg・2026-10-10・本人の決定）。

- 機械音 A `react_thinking.wav`：Jev が判断しているあいだ鳴らし続ける。600 Hz の短い「ポッ」（0.08 秒）＋休みで 1 周期
  0.35 秒。鳴らす側（`io/dif`）が繰り返す。
- 効果音 B `react_ack.wav`：軽く返す・聞き返すときに 1 回。880 Hz → 1320 Hz と上がる「ピロッ」（0.12 秒）。
- 作り置きの声 `react_tool_*.wav`（はい・うん・りょ）・`react_think_*.wav`（んー・えっと・うーん）：いまの声
  （Style-Bert-VITS2・`.env` の `SBV2_*`）で読み上げる。合成サーバーが動いていなければ起こし、作り終えたら止める。

音量はタイマー（振幅 0.6）よりずっと小さくする（本人：もう一声小さく）。A は 0.15、B は 0.25。拾ってきた音源は許諾の確認と
出典の記録が要るので、効果音は自分で合成する（`gen_timer_alarm.py` と同じ）。

使い方：`uv run python scripts/gen_reaction_sounds.py`（効果音だけなら `--tones`）
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "src")

from familiar_agent.core import reaction_cue as rc  # noqa: E402

RATE = 48000  # 再生側は機器の rate へ合わせ直すが、元もタイマーと揃えておく
THINK_FREQ = 600.0
THINK_BEEP_SEC = 0.08
THINK_PERIOD_SEC = 0.35
THINK_AMPLITUDE = 0.15
ACK_FREQS = (880.0, 1320.0)  # 上がる 2 音
ACK_NOTE_SEC = 0.06
ACK_AMPLITUDE = 0.25


def _tone(freq: float, sec: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(RATE * sec)) / RATE
    wave_ = np.sin(2 * np.pi * freq * t)
    ramp = int(RATE * 0.005)  # 頭と尻を 5 ms で絞り、クリック音を出さない
    env = np.ones_like(wave_)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return wave_ * env * amplitude


def build_thinking() -> np.ndarray:
    """機械音 A の 1 周期（ポッ ＋ 休み）。"""
    beep = _tone(THINK_FREQ, THINK_BEEP_SEC, THINK_AMPLITUDE)
    rest = np.zeros(max(0, int(RATE * THINK_PERIOD_SEC) - len(beep)))
    return np.concatenate([beep, rest]).astype(np.float32)


def build_ack() -> np.ndarray:
    """効果音 B（ピロッ）。"""
    return np.concatenate([_tone(f, ACK_NOTE_SEC, ACK_AMPLITUDE) for f in ACK_FREQS]).astype(
        np.float32
    )


def _write(path: pathlib.Path, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def write_tones() -> None:
    _write(rc.THINKING_WAV, build_thinking())
    _write(rc.ACK_WAV, build_ack())
    print(f"{rc.THINKING_WAV.name}・{rc.ACK_WAV.name}")


def write_voices() -> int:
    import json
    import urllib.request

    # `.env` を自前で読む（この道具は familiar の起動経路を通らない）。
    for line in pathlib.Path(".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from familiar_agent.config import TTSConfig
    from familiar_agent.tools import tts

    cfg = TTSConfig()
    started_here = False
    if not tts._sbv2_is_alive(cfg.sbv2_url):
        tts._spawn_sbv2(cfg)
        started_here = True
        for _ in range(120):
            if tts._sbv2_is_alive(cfg.sbv2_url):
                break
            time.sleep(1)
        else:
            print("合成サーバーが 120 秒で起きなかった。")
            return 1
    try:
        pairs = list(zip(rc.TOOL_WORDS, rc.TOOL_WAVS)) + list(zip(rc.THINK_WORDS, rc.THINK_WAVS))
        for text, path in pairs:
            payload = json.dumps(
                {"text": text, "style": cfg.sbv2_style, "weight": cfg.sbv2_weight}
            ).encode()
            req = urllib.request.Request(
                f"{cfg.sbv2_url}/synth", data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                wav = resp.read()
            path.write_bytes(wav)
            print(f"{path.name}：{text}（{len(wav)} バイト）")
    finally:
        if started_here and tts._sbv2_proc is not None:
            tts._sbv2_proc.terminate()
    return 0


def main() -> int:
    write_tones()
    if "--tones" in sys.argv:
        return 0
    return write_voices()


if __name__ == "__main__":
    sys.exit(main())
