#!/usr/bin/env python
"""話しかけたときの合図の音 `src/familiar_agent/sounds/react_*.wav` を作る（出-bg・2026-10-10・本人の決定）。

音の高さは、ドを 523.25 Hz に置いた純正律のハ長調の 1 つの音階から取る（環-aw 第 1 段 項目 6・2026-11-09・本人が聞いて
決めた）。一瞬で終わる段は和音、時間がかかる段は単音にし、段から段へ高さがつながるようにした。大きさは、振幅 0.15 を基に
した dB で決めた（黙る音だけは 0.35 を基にする）。

| 音 | ファイル | 高さ | 形 | 長さ | 大きさ |
|---|---|---|---|---|---|
| 思い出している | `react_beep_recalling.wav` | ド・ミ・ソ | だんだん消える | 0.40 秒 | −1 dB |
| 機械音 A・決めている | `react_thinking.wav` | ド | 平ら | 0.08 秒（0.35 秒で 1 周期） | −6 dB |
| 返りを待っている | `react_beep_awaiting.wav` | ミ | 平ら | 0.06 秒 | −6 dB |
| 確かめている・効果音 B | `react_beep_checking.wav`・`react_ack.wav` | ド・ミ・ソ | 平ら | 0.12 秒 | −1 dB |
| 書き留めている | `react_beep_noting.wav` | 1 つ下のド | 平ら | 0.12 秒 | −24 dB |
| 黙る音 | `react_silent.wav` | 1 つ下のド・ミ・ソ → さらに低いラ・ド・ミ | だんだん消える | 0.15 秒 → 0.40 秒 | −6 dB（0.35 から） |

- 「平ら」は、5 ms で立ち上がり、長さの 15 % ＋ 5 ms で絞って終わる。「だんだん消える」は、叩いた音のように鳴らした瞬間が
  いちばん大きく、長さの 4 分の 1 ごとに 1/e へ減る（立ち上がりは 5 ms）。
- 機械音 A は Jev が判断しているあいだ、鳴らす側（`io/dif`）が繰り返す。考えているの子の電子音は、待ちのループが
  1 秒ごとに 1 回鳴らす（`wait_beep_interval_seconds`）。
- 作り置きの声 `react_tool_0.wav`（はいっ）・`react_think_0.wav`（えっと）：いまの声（Style-Bert-VITS2・`.env` の
  `SBV2_*`）で読み上げる。合成サーバーが動いていなければ起こし、作り終えたら止める（出-de・2026-10-31・本人：最初の反応は
  この 2 つだけ）。「はいっ」は作ったあと、出だしを `VOICE_ONSET_SEC` かけて丸くする（やわらかくする・本人が聞いて決めた）。
- 大きさを揃える（出-df・2026-11-01）：声を作り終えたら、そのフォルダの 2 つの声を、声が出ている部分の実効値（`active_db`）で
  `VOICE_LEVEL_DB` に合わせる（環-aw 第 1 段 項目 6 で、「えっと」に合わせる形から決まった大きさに合わせる形にした）。
  大きくするときは最大の振れを `PEAK_LIMIT_DB` で止める。声を作らずに揃えるだけなら `--level`（元の声と、声ごとのフォルダ全部）。
- 時間切れの一言 `timeout_answer.wav`（「時間内に答えをまとめられませんでした。」・環-aw 第 1 段 項目 8）：時間切れの答えを
  軽量LLM が書けなかったときに鳴らす。ほかの作り置きの声と同じに作り、同じ大きさに揃える。

拾ってきた音源は許諾の確認と出典の記録が要るので、効果音は自分で合成する（`gen_timer_alarm.py` と同じ）。

使い方：`uv run python scripts/gen_reaction_sounds.py`（効果音だけなら `--tones`。言葉を選んで作るなら `--only はいっ`。
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
#: 音階の基のド（Hz）と、ドを 1 とした純正律の比（環-aw 第 1 段 項目 6）。全部の音をこの 1 つの音階から取る。
C5 = 523.25
JUST = {"ド": 1.0, "レ": 9 / 8, "ミ": 5 / 4, "ファ": 4 / 3, "ソ": 3 / 2, "ラ": 5 / 3, "シ": 15 / 8}
#: 大きさの基の振幅と、音ごとの dB。和音は −1 dB、単音は −6 dB、書き留めるは −24 dB（本人が聞いて決めた）。
BASE_AMPLITUDE = 0.15
CHORD_DB = -1.0
SINGLE_DB = -6.0
NOTING_DB = -24.0
#: 機械音 A の 1 周期。
THINK_PERIOD_SEC = 0.35
#: 黙る音（出-dc）：振幅 0.35 を基にして −6 dB。
SILENT_AMPLITUDE = 0.35
SILENT_DB = -6.0
#: 考えているの子のうち、自分の音のファイルを持つもの（決めているは機械音 A のファイルを使う）。表は `reaction_cue` が持つ。
BEEP_KINDS = tuple(k for k, p in rc.BEEP_WAVS.items() if p != rc.THINKING_WAV)
#: 作り置きの声の大きさ（dB・声が出ている部分の実効値）。いまの「えっと」（−20.1 dB）から −6 dB（本人が聞いて決めた）。
VOICE_LEVEL_DB = -26.1
#: 道具の声の出だしを丸くする長さ（秒）。声が 0.01 を超えた所から上げ、その前は消す。
VOICE_ONSET_SEC = 0.12
#: 大きさを測る窓（秒）と、数に入れる窓（いちばん大きい窓から何 dB 以内か）。無音と息の音を数に入れない（出-df）。
LEVEL_WINDOW_SEC = 0.02
LEVEL_FLOOR_DB = 30.0
#: 大きくするときの最大の振れの上限（dBFS・出-df・2026-11-01・本人の承認）。超えると音が割れる。
PEAK_LIMIT_DB = -1.0
#: 変える量がこれより小さければ書き直さない（dB）。作り直すたびに丸めの誤差が積もらないようにする。
LEVEL_SKIP_DB = 0.05


def pitch(name: str, octave: int = 0) -> float:
    """音階の音（Hz）。`octave` が −1 なら 1 つ下。"""
    return C5 * JUST[name] * 2.0**octave


def _amp(db: float, base: float = BASE_AMPLITUDE) -> float:
    return base * 10 ** (db / 20)


def _mix(freqs: "tuple[float, ...]", sec: float) -> "tuple[np.ndarray, np.ndarray]":
    t = np.arange(int(RATE * sec)) / RATE
    return t, sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)


def _flat(freqs: "tuple[float, ...]", sec: float, amplitude: float) -> np.ndarray:
    """平らに鳴る音。5 ms で立ち上がり、長さの 15 % ＋ 5 ms で絞る（ぶつっと切れない）。"""
    _, w = _mix(freqs, sec)
    a, r = int(RATE * 0.005), int(RATE * (0.005 + sec * 0.15))
    env = np.ones_like(w)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r)
    return w * env * amplitude


def _decay(freqs: "tuple[float, ...]", sec: float, amplitude: float) -> np.ndarray:
    """だんだん消える音。鳴らした瞬間がいちばん大きく、長さの 4 分の 1 ごとに 1/e へ減る（立ち上がりは 5 ms）。"""
    t, w = _mix(freqs, sec)
    a = int(RATE * 0.005)
    env = np.exp(-t / (sec / 4))
    env[:a] *= np.linspace(0.0, 1.0, a)
    return w * env * amplitude


_TRIAD = (pitch("ド"), pitch("ミ"), pitch("ソ"))


def build_thinking() -> np.ndarray:
    """機械音 A の 1 周期（ドを 0.08 秒 ＋ 休み）。決めているの子も同じ音。"""
    beep = _flat((pitch("ド"),), 0.08, _amp(SINGLE_DB))
    rest = np.zeros(max(0, int(RATE * THINK_PERIOD_SEC) - len(beep)))
    return np.concatenate([beep, rest]).astype(np.float32)


def build_beep(kind: str) -> np.ndarray:
    """考えているの子の電子音（1 回ぶん。間隔は待ちのループがあける）。"""
    if kind == "recalling":
        x = _decay(_TRIAD, 0.40, _amp(CHORD_DB))
    elif kind == "awaiting":
        x = _flat((pitch("ミ"),), 0.06, _amp(SINGLE_DB))
    elif kind == "checking":
        x = _flat(_TRIAD, 0.12, _amp(CHORD_DB))
    elif kind == "noting":
        x = _flat((pitch("ド", -1),), 0.12, _amp(NOTING_DB))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def build_ack() -> np.ndarray:
    """効果音 B。確かめているの子と同じ音（ド・ミ・ソを 0.12 秒）。"""
    return build_beep("checking")


def build_silent() -> np.ndarray:
    """黙る音。1 つ下のド・ミ・ソを 0.15 秒、続けてさらに低いラ・ド・ミを 0.40 秒。どちらもだんだん消える。"""
    amp = _amp(SILENT_DB, SILENT_AMPLITUDE)
    high = _decay((pitch("ド", -1), pitch("ミ", -1), pitch("ソ", -1)), 0.15, amp)
    low = _decay((pitch("ラ", -3), pitch("ド", -2), pitch("ミ", -2)), 0.40, amp)
    return np.concatenate([high, low]).astype(np.float32)


def soft_onset(samples: np.ndarray, rate: int, sec: float = VOICE_ONSET_SEC) -> np.ndarray:
    """声の出だしを丸くする。声が 0.01 を超えた所から `sec` 秒かけて上げ、その前は消す。声が無ければそのまま。"""
    loud = np.flatnonzero(np.abs(samples) > 0.01)
    if len(loud) == 0:
        return samples
    on, n = int(loud[0]), int(rate * sec)
    y = np.array(samples, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    y[on : on + n] *= ramp[: len(y) - on]
    y[:on] = 0.0
    return y


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


def _rewrite(path: pathlib.Path, samples: np.ndarray, params: "wave._wave_params") -> None:
    pcm = np.round(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(pcm.tobytes())


def level_file(path: pathlib.Path) -> None:
    """`path` の声を `VOICE_LEVEL_DB` の大きさにする。

    無音（大きさが測れない）・形が違うときは、書き換えずに知らせる。変える量がわずかなら書き直さない。
    """
    label = f"{path.parent.name}/{path.name}"
    try:
        x, params = _read_wav(path)
    except ValueError as e:
        print(f"{label}：揃えなかった（{e}）")
        return
    before = active_db(x, params.framerate)
    if not np.isfinite(before):
        print(f"{label}：揃えなかった（無音）")
        return
    if abs(VOICE_LEVEL_DB - before) < LEVEL_SKIP_DB:
        print(f"{label}：{before:.1f} dB のまま")
        return
    out, short = match_level(x, VOICE_LEVEL_DB, params.framerate)
    _rewrite(path, out, params)
    note = f"・{short:.1f} dB 足りない（最大の振れの上限）" if short else ""
    print(f"{label}：{before:.1f} → {active_db(out, params.framerate):.1f} dB{note}")


def level_folder(folder: pathlib.Path) -> None:
    """`folder` の作り置きの声（道具の声と考える声）を、どちらも `VOICE_LEVEL_DB` の大きさにする（出-df）。"""
    for path in (folder / p.name for p in (*rc.TOOL_WAVS, *rc.THINK_WAVS, rc.TIMEOUT_WAV)):
        if path.exists():
            level_file(path)


def soften_file(path: pathlib.Path) -> None:
    """作ったばかりの道具の声の出だしを丸くする（`soft_onset`）。作るたびに 1 回だけかける。"""
    x, params = _read_wav(path)
    _rewrite(path, soft_onset(x, params.framerate), params)


def _all_folders() -> "list[pathlib.Path]":
    base = rc.TOOL_WAVS[0].parent
    voices = base / "voices"
    return [base, *sorted(p for p in voices.iterdir() if p.is_dir())] if voices.exists() else [base]


def _pairs(only: "set[str] | None") -> "list[tuple[str, pathlib.Path]]":
    """作る言葉と置き場の組。`only` があれば、その言葉だけ（聞いて決めたほかの声を作り直さない・出-ce）。"""
    pairs = list(zip(rc.TOOL_WORDS, rc.TOOL_WAVS)) + list(zip(rc.THINK_WORDS, rc.THINK_WAVS))
    pairs.append((rc.TIMEOUT_WORDS, rc.TIMEOUT_WAV))
    return [(t, p) for t, p in pairs if only is None or t in only]


def _only() -> "set[str] | None":
    """`--only 言葉,言葉` の言葉。無ければ None（全部）。"""
    if "--only" not in sys.argv:
        return None
    return {w.strip() for w in sys.argv[sys.argv.index("--only") + 1].split(",") if w.strip()}


def write_tones() -> None:
    made = {
        rc.THINKING_WAV: build_thinking(),
        rc.ACK_WAV: build_ack(),
        rc.SILENT_WAV: build_silent(),
    }
    made.update({rc.BEEP_WAVS[k]: build_beep(k) for k in BEEP_KINDS})
    for path, data in made.items():
        _write(path, data)
    print("・".join(p.name for p in made))


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
            if text in rc.TOOL_WORDS:
                soften_file(path)
            print(f"{path.name}：{text}（{len(wav)} バイト）")
        level_folder(rc.TOOL_WAVS[0].parent)  # 作り直しても決まった大きさのままにする（出-df）
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
        if text in rc.TOOL_WORDS:
            soften_file(path)
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
