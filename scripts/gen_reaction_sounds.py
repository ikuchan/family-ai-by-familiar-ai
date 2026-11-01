#!/usr/bin/env python
"""話しかけたときの合図の音 `src/familiar_agent/sounds/react_*.wav` を作る（出-bg・2026-10-10・本人の決定）。

- 機械音 A `react_thinking.wav`：Jev が判断しているあいだ鳴らし続ける。600 Hz の短い「ポッ」（0.08 秒）＋休みで 1 周期
  0.35 秒。鳴らす側（`io/dif`）が繰り返す。
- 効果音 B `react_ack.wav`：軽く返す・聞き返すときに 1 回。880 Hz → 1320 Hz と上がる「ピロッ」（0.12 秒）。
- 黙る音 `react_silent.wav`：合図の後で黙ると決まったときに 1 回（出-dc・2026-10-26・本人が聞いて決めた）。低い和音
  （ド・ミ・ソ）を 0.4 秒、出だしと終わりを丸くやわらかく。振幅 0.35。`--tones` と既定の実行で A・B と一緒に書き直す。
- 作り置きの声 `react_tool_0.wav`（はーい！）・`react_think_0.wav`（えっと）：いまの声（Style-Bert-VITS2・`.env` の
  `SBV2_*`）で読み上げる。合成サーバーが動いていなければ起こし、作り終えたら止める（出-de・2026-10-31・本人：最初の反応は
  この 2 つだけ）。
- 大きさを揃える（出-df・2026-11-01・本人の決定ア）：声を作り終えたら、そのフォルダの「はーい！」を「えっと」と同じ大きさに
  する（声が出ている部分の実効値・`active_db`）。大きくするときは最大の振れを `PEAK_LIMIT_DB` で止める。声を作らずに
  揃えるだけなら `--level`（元の声と、声ごとのフォルダ全部）。

音量はタイマー（振幅 0.6）よりずっと小さくする（本人：もう一声小さく）。A は 0.15、B は 0.25。黙る音は 0.35（和音なので
平均の強さは B の約 7 割・本人が聞いて決めた値）。拾ってきた音源は許諾の確認と
出典の記録が要るので、効果音は自分で合成する（`gen_timer_alarm.py` と同じ）。

使い方：`uv run python scripts/gen_reaction_sounds.py`（効果音だけなら `--tones`。言葉を選んで作るなら `--only はーい！`。
揃えるだけなら `--level`）

声ごとの作り置き（出-bv）：`--voice Laomedeia` のように声を指定すると、作り置きの声だけを Gemini のその声で読み上げ、
`sounds/voices/<声>/` に書く（返事と声をそろえる。鳴らす側は `.env` の `GEMINI_TTS_VOICE` の声のフォルダから選ぶ）。
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
#: 黙る音（出-dc・2026-10-26・本人が YAMAHA で聞いて決めた）：低い和音（ド・ミ・ソ）を 0.4 秒。出だし 0.06 秒・終わり 0.18 秒を
#: 丸い曲線でやわらかくする。振幅 0.35 も聞いて決めた値（B に連動させない。3 音を 3 で割るので、平均の強さは B の約 7 割）。
SILENT_FREQS = (130.81, 164.81, 196.00)
SILENT_SEC = 0.4
SILENT_ATTACK_SEC = 0.06
SILENT_RELEASE_SEC = 0.18
SILENT_AMPLITUDE = 0.35
#: 大きさを測る窓（秒）と、数に入れる窓（いちばん大きい窓から何 dB 以内か）。無音と息の音を数に入れない（出-df）。
LEVEL_WINDOW_SEC = 0.02
LEVEL_FLOOR_DB = 30.0
#: 大きくするときの最大の振れの上限（dBFS・出-df・2026-11-01・本人の承認）。超えると音が割れる。
PEAK_LIMIT_DB = -1.0
#: 変える量がこれより小さければ書き直さない（dB）。作り直すたびに丸めの誤差が積もらないようにする。
LEVEL_SKIP_DB = 0.05


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


def build_silent() -> np.ndarray:
    """黙る音（低い和音・出だしと終わりを丸くやわらかく）。"""
    t = np.arange(int(RATE * SILENT_SEC)) / RATE
    wave_ = sum(np.sin(2 * np.pi * f * t) for f in SILENT_FREQS) / len(SILENT_FREQS)
    env = np.ones_like(t)
    a, r = int(RATE * SILENT_ATTACK_SEC), int(RATE * SILENT_RELEASE_SEC)
    env[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, a))
    env[-r:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, r))
    return (wave_ * env * SILENT_AMPLITUDE).astype(np.float32)


def _write(path: pathlib.Path, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def active_db(samples: np.ndarray, rate: int = 24000) -> float:
    """声が出ている部分の大きさ（dB・実効値）。20 ms の窓のうち、いちばん大きい窓から 30 dB 以内の窓だけを数える。"""
    n = max(1, int(rate * LEVEL_WINDOW_SEC))
    k = len(samples) // n
    if k == 0:
        return float("-inf")
    win = np.sqrt((np.asarray(samples[: k * n], dtype=np.float64).reshape(k, n) ** 2).mean(axis=1))
    top = float(win.max())
    if top <= 0.0:
        return float("-inf")
    kept = win[win >= top * 10 ** (-LEVEL_FLOOR_DB / 20)]
    return 20 * float(np.log10(np.sqrt((kept**2).mean())))


def match_level(
    samples: np.ndarray, target_db: float, rate: int = 24000
) -> "tuple[np.ndarray, float]":
    """`target_db` の大きさに合わせる。大きくするときは最大の振れを `PEAK_LIMIT_DB` で止め、足りなかった dB を返す。"""
    gain_db = target_db - active_db(samples, rate)
    peak = float(np.abs(samples).max())
    if gain_db > 0 and peak > 0:
        room_db = PEAK_LIMIT_DB - 20 * float(np.log10(peak))
        if gain_db > room_db:
            short = gain_db - max(room_db, 0.0)
            gain_db = max(room_db, 0.0)
            return samples * 10 ** (gain_db / 20), short
    return samples * 10 ** (gain_db / 20), 0.0


def _read_wav(path: pathlib.Path) -> "tuple[np.ndarray, wave._wave_params]":
    """1 チャンネル・16 ビットの wav だけ読む（作り置きの声はどれもこの形）。違えば ValueError。"""
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        if params.nchannels != 1 or params.sampwidth != 2:
            raise ValueError(f"{path.name} は 1 チャンネル・16 ビットでない")
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
    return data, params


def level_folder(folder: pathlib.Path) -> None:
    """`folder` の「はーい！」（`react_tool_0.wav`）を、同じフォルダの「えっと」（`react_think_0.wav`）の大きさに揃える。

    どちらかが無音（大きさが測れない）・形が違うときは、書き換えずに知らせる。変える量がわずかなら書き直さない。
    """
    hai_path = folder / rc.TOOL_WAVS[0].name
    eto_path = folder / rc.THINK_WAVS[0].name
    if not (hai_path.exists() and eto_path.exists()):
        return
    try:
        eto, eto_params = _read_wav(eto_path)
        hai, params = _read_wav(hai_path)
    except ValueError as e:
        print(f"{folder.name}：揃えなかった（{e}）")
        return
    target = active_db(eto, eto_params.framerate)
    before = active_db(hai, params.framerate)
    if not (np.isfinite(target) and np.isfinite(before)):
        print(f"{folder.name}：揃えなかった（無音のファイルがある）")
        return
    if abs(target - before) < LEVEL_SKIP_DB:
        print(f"{folder.name}/{hai_path.name}：{before:.1f} dB（えっと {target:.1f}）のまま")
        return
    out, short = match_level(hai, target, params.framerate)
    pcm = np.round(np.clip(out, -1.0, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(hai_path), "wb") as w:
        w.setparams(params)
        w.writeframes(pcm.tobytes())
    after = active_db(out, params.framerate)
    note = f"・{short:.1f} dB 足りない（最大の振れの上限）" if short else ""
    print(
        f"{folder.name}/{hai_path.name}：{before:.1f} → {after:.1f} dB（えっと {target:.1f}）{note}"
    )


def _all_folders() -> "list[pathlib.Path]":
    base = rc.TOOL_WAVS[0].parent
    voices = base / "voices"
    return [base, *sorted(p for p in voices.iterdir() if p.is_dir())] if voices.exists() else [base]


def _pairs(only: "set[str] | None") -> "list[tuple[str, pathlib.Path]]":
    """作る言葉と置き場の組。`only` があれば、その言葉だけ（聞いて決めたほかの声を作り直さない・出-ce）。"""
    pairs = list(zip(rc.TOOL_WORDS, rc.TOOL_WAVS)) + list(zip(rc.THINK_WORDS, rc.THINK_WAVS))
    return [(t, p) for t, p in pairs if only is None or t in only]


def _only() -> "set[str] | None":
    """`--only 言葉,言葉` の言葉。無ければ None（全部）。"""
    if "--only" not in sys.argv:
        return None
    return {w.strip() for w in sys.argv[sys.argv.index("--only") + 1].split(",") if w.strip()}


def write_tones() -> None:
    _write(rc.THINKING_WAV, build_thinking())
    _write(rc.ACK_WAV, build_ack())
    _write(rc.SILENT_WAV, build_silent())
    print(f"{rc.THINKING_WAV.name}・{rc.ACK_WAV.name}・{rc.SILENT_WAV.name}")


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
        for text, path in _pairs(_only()):
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
        level_folder(
            rc.TOOL_WAVS[0].parent
        )  # 作り直しても「はーい！」を「えっと」に揃えたままにする（出-df）
    finally:
        if started_here and tts._sbv2_proc is not None:
            tts._sbv2_proc.terminate()
    return 0


def _load_env() -> None:
    for line in pathlib.Path(".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def write_gemini_voices(voice: str) -> int:
    """作り置きの声を Gemini の `voice` で読み上げ、`sounds/voices/<voice>/` に書く（出-bv）。"""
    _load_env()
    from familiar_agent.config import TTSConfig
    from familiar_agent.tools import tts

    cfg = TTSConfig()
    for text, base in _pairs(_only()):
        path = base.parent / "voices" / voice / base.name
        path.parent.mkdir(parents=True, exist_ok=True)
        wav = tts.gemini_wav(cfg, text, voice)
        path.write_bytes(wav)
        print(f"{voice}/{path.name}：{text}（{len(wav)} バイト）")
        time.sleep(7)  # 無料枠は 1 分に 10 回
    level_folder(rc.TOOL_WAVS[0].parent / "voices" / voice)  # 作り直しても揃えたままにする（出-df）
    return 0


def main() -> int:
    if "--level" in sys.argv:
        for folder in _all_folders():
            level_folder(folder)
        return 0
    if "--voice" in sys.argv:
        return write_gemini_voices(sys.argv[sys.argv.index("--voice") + 1])
    if "--only" not in sys.argv:
        write_tones()
    if "--tones" in sys.argv:
        return 0
    return write_voices()


if __name__ == "__main__":
    sys.exit(main())
