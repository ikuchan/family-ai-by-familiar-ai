"""合図の音を作る道具（`scripts/gen_reaction_sounds.py`）。

- 作り置きの声を、指定した言葉だけ作る（`--only`・出-ce・2026-10-14）。全部を作り直すと、聞いて決めた声まで変わる。
- 声の大きさを揃える（出-df・2026-11-01）。環-aw 第 1 段 項目 6（2026-11-09・本人）で、「えっと」に合わせる形から、
  決まった大きさ（`VOICE_LEVEL_DB`）に「はいっ」と「えっと」の両方を合わせる形にした。
- 音の高さ・長さ・大きさ（環-aw 第 1 段 項目 6）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_reaction_sounds", Path("scripts/gen_reaction_sounds.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_only_the_named_words_are_made():
    gen = _script()
    got = [text for text, _ in gen._pairs({"はいっ"})]
    assert got == ["はいっ"]


def test_without_only_every_word_is_made():
    """**反証の側**：指定が無ければ、今までどおり全部。"""
    gen = _script()
    got = [text for text, _ in gen._pairs(None)]
    assert got == ["はいっ", "えっと"] and len(got) == len(gen.rc.TOOL_WORDS) + len(
        gen.rc.THINK_WORDS
    )


# ── 声の大きさを揃える（出-df・2026-11-01。環-aw 第 1 段 項目 6 で決まった大きさに合わせる形にした） ─────────────

import math  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402


def _sine(amplitude: float, sec: float = 0.5, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(rate * sec)) / rate
    return amplitude * np.sin(2 * np.pi * 440.0 * t)


def test_the_level_of_a_sine_is_measured_without_the_silence():
    gen = _script()
    tone = _sine(0.5)
    expected = 20 * math.log10(0.5 / math.sqrt(2))  # 約 −9.0 dB
    assert abs(gen.active_db(tone) - expected) < 0.1
    padded = np.concatenate([np.zeros(24000), tone, np.zeros(48000)])
    assert (
        abs(gen.active_db(padded) - gen.active_db(tone)) < 0.1
    )  # 反証の側：無音の長さで変わらない


def test_a_louder_voice_is_brought_down_to_the_target():
    gen = _script()
    out, short = gen.match_level(_sine(0.8), -20.0)
    assert abs(gen.active_db(out) - -20.0) < 0.1 and short == 0.0
    same, _ = gen.match_level(_sine(0.8), gen.active_db(_sine(0.8)))
    assert np.allclose(same, _sine(0.8))  # 反証の側：目標と同じなら変えない


def test_raising_stops_at_the_peak_limit_and_reports_the_shortfall():
    gen = _script()
    out, short = gen.match_level(
        _sine(0.5), -1.0
    )  # 正弦波で実効値 −1 dB は最大の振れが 0 dBFS を超える
    assert 20 * math.log10(np.abs(out).max()) <= gen.PEAK_LIMIT_DB + 1e-6
    assert short > 0.0
    ok, short2 = gen.match_level(_sine(0.1), -15.0)  # 反証の側：余裕があれば目標に合う
    assert abs(gen.active_db(ok) - -15.0) < 0.1 and short2 == 0.0


def _write(path: Path, data: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())


def _read(path: Path) -> "tuple[np.ndarray, tuple[int, int, int]]":
    with wave.open(str(path), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
    return data, fmt


def test_a_silent_file_is_left_untouched(tmp_path):
    """無音のファイルは、大きさが測れないので書き換えない。"""
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.8), 24000)
    _write(tmp_path / "react_think_0.wav", np.zeros(12000), 24000)
    before = (tmp_path / "react_think_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    assert (tmp_path / "react_think_0.wav").read_bytes() == before
    x, _ = _read(tmp_path / "react_tool_0.wav")
    assert abs(gen.active_db(x) - gen.VOICE_LEVEL_DB) < 0.1  # 反証の側：測れるほうは揃える


def test_a_matched_file_is_not_rewritten(tmp_path):
    """もう揃っていれば書き直さない（作り直すたびに丸めの誤差を積もらせない）。"""
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.3), 24000)
    gen.level_folder(tmp_path)
    once = (tmp_path / "react_tool_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    assert (tmp_path / "react_tool_0.wav").read_bytes() == once


def test_a_stereo_file_is_refused(tmp_path):
    gen = _script()
    with wave.open(str(tmp_path / "react_tool_0.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.zeros(4800) * 0).astype("<i2").tobytes())
    _write(tmp_path / "react_think_0.wav", _sine(0.3), 24000)
    before = (tmp_path / "react_tool_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    assert (tmp_path / "react_tool_0.wav").read_bytes() == before


def test_the_level_is_measured_at_the_file_rate(tmp_path):
    """44.1 kHz の元の声でも、ファイルの速さで測って合わせる。"""
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.8, rate=44100), 44100)
    _write(tmp_path / "react_think_0.wav", _sine(0.1, rate=44100), 44100)
    gen.level_folder(tmp_path)
    for name in ("react_tool_0.wav", "react_think_0.wav"):
        x, _ = _read(tmp_path / name)
        assert abs(gen.active_db(x, 44100) - gen.VOICE_LEVEL_DB) < 0.05, name


# ── 環-aw 第 1 段 項目 6：音はドを 523.25 Hz に置いた純正律から取る（2026-11-09・本人が聞いて決めた） ─────────────
#
# 一瞬で終わる段は和音、時間がかかる段は単音にし、高さをつなげる。大きさは 0.15 を基にした dB で決めた。


def _peaks(x: np.ndarray, rate: int, n: int) -> "list[float]":
    """大きい順に `n` 個の周波数（Hz）。窓をかけ、0 を詰めて細かく測る。"""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 20))
    freqs = np.fft.rfftfreq(1 << 20, 1 / rate)
    got: list = []
    for i in np.argsort(spec)[::-1]:
        if all(abs(freqs[i] - f) > 20 for f in got):
            got.append(float(freqs[i]))
        if len(got) == n:
            break
    return sorted(got)


C5 = 523.25


def _close(got, want, tol=1.0):
    return len(got) == len(want) and all(abs(g - w) < tol for g, w in zip(got, want))


def test_the_scale_is_just_intonation_on_c5():
    gen = _script()
    assert gen.pitch("ド") == C5
    assert gen.pitch("ミ") == C5 * 5 / 4 and gen.pitch("ソ") == C5 * 3 / 2
    assert gen.pitch("ラ", -3) == C5 * 5 / 3 / 8  # 110 Hz
    assert gen.pitch("ド", -1) == C5 / 2


def test_each_step_has_its_pitch():
    gen = _script()
    r = gen.RATE
    assert _close(_peaks(gen.build_thinking(), r, 1), [C5])  # 機械音 A・決めている：ド
    assert _close(_peaks(gen.build_beep("awaiting"), r, 1), [C5 * 5 / 4])  # 返りを待つ：ミ
    assert _close(_peaks(gen.build_beep("noting"), r, 1), [C5 / 2])  # 書き留める：低いド
    triad = [C5, C5 * 5 / 4, C5 * 3 / 2]
    assert _close(_peaks(gen.build_beep("recalling"), r, 3), triad)  # 思い出す：ドミソ
    assert _close(_peaks(gen.build_beep("checking"), r, 3), triad)  # 確かめる：ドミソ
    assert _close(_peaks(gen.build_ack(), r, 3), triad)  # 効果音 B：確かめると同じ
    assert not _close(_peaks(gen.build_thinking(), r, 1), [600.0])  # 反証の側：前の 600 Hz ではない


def test_each_step_has_its_length_and_loudness():
    gen = _script()
    r = gen.RATE

    def db(x):
        return 20 * math.log10(np.abs(x).max() / 0.15)

    def sounding(x):  # 鳴っている長さ（最後に 0 でない標本まで）
        return (np.flatnonzero(np.abs(x) > 0)[-1] + 1) / r

    a = gen.build_thinking()
    assert abs(len(a) / r - 0.35) < 0.001 and abs(sounding(a) - 0.08) < 0.001
    for kind, sec, loud in (
        ("recalling", 0.40, -1.0),
        ("awaiting", 0.06, -6.0),
        ("checking", 0.12, -1.0),
        ("noting", 0.12, -24.0),
    ):
        x = gen.build_beep(kind)
        assert abs(len(x) / r - sec) < 0.001, kind
        # 単音は最大の振れが振幅に届く。和音は 3 つを混ぜるので届かず、振幅を超えないことだけを見る
        if kind in ("awaiting", "noting"):
            assert abs(db(x) - loud) < 0.1, kind
        else:
            assert db(x) <= loud + 1e-6, kind
    assert (gen.CHORD_DB, gen.SINGLE_DB, gen.NOTING_DB) == (-1.0, -6.0, -24.0)
    assert abs(db(a) - -6.0) < 0.1
    assert np.array_equal(gen.build_ack(), gen.build_beep("checking"))


def test_recalling_fades_and_checking_stays_flat():
    """思い出すは鳴らした瞬間がいちばん大きく、だんだん消える。確かめるは平らに鳴って、終わりで絞る。"""
    gen = _script()
    r = gen.RATE

    def rms(x, a, b):
        return float(np.sqrt((x[int(a * r) : int(b * r)] ** 2).mean()))

    rec = gen.build_beep("recalling")
    assert rms(rec, 0.30, 0.35) < rms(rec, 0.01, 0.06) / 5
    chk = gen.build_beep("checking")
    assert abs(rms(chk, 0.07, 0.09) - rms(chk, 0.01, 0.03)) < rms(chk, 0.01, 0.03) * 0.1


def test_the_silent_sound_is_two_falling_chords():
    """黙る音：ド・ミ・ソ（下のオクターブ）を 0.15 秒、続けて ラ・ド・ミ（さらに低く）を 0.40 秒。どちらもだんだん消える。"""
    gen = _script()
    r = gen.RATE
    x = gen.build_silent()
    assert abs(len(x) / r - 0.55) < 0.001
    assert _close(_peaks(x[: int(0.15 * r)], r, 3), [C5 / 2, C5 * 5 / 8, C5 * 3 / 4], tol=3.0)
    assert _close(_peaks(x[int(0.15 * r) :], r, 3), [110.0, C5 / 4, C5 * 5 / 16], tol=1.0)
    assert np.abs(x).max() <= 0.35 * 10 ** (-6 / 20) + 1e-6
    assert (gen.SILENT_AMPLITUDE, gen.SILENT_DB) == (0.35, -6.0)


def test_the_files_are_what_the_tool_makes():
    """置いてある音は、道具がいまの値で作るものと同じ（作り直せる）。"""
    gen = _script()
    made = {
        gen.rc.THINKING_WAV: gen.build_thinking(),
        gen.rc.ACK_WAV: gen.build_ack(),
        gen.rc.SILENT_WAV: gen.build_silent(),
        **{gen.rc.BEEP_WAVS[k]: gen.build_beep(k) for k in gen.BEEP_KINDS},
    }
    for path, x in made.items():
        data, fmt = _read(path)
        assert fmt == (1, 2, gen.RATE), path.name
        assert len(data) == len(x) and np.abs(data - x).max() < 2 / 32768, path.name


def test_a_soft_onset_rounds_only_the_start():
    gen = _script()
    rate = 24000
    x = np.concatenate([np.full(240, 0.001), _sine(0.5, 0.5, rate)])
    y = gen.soft_onset(x, rate)
    on = int(np.flatnonzero(np.abs(x) > 0.01)[0])
    assert not y[:on].any()  # 出だしより前は消す
    n = int(rate * gen.VOICE_ONSET_SEC)
    assert np.abs(y[on : on + n // 10]).max() < 0.05  # 立ち上がりは小さい
    assert np.array_equal(y[on + n :], x[on + n :])  # 反証の側：その先は変えない


def test_both_words_are_leveled_to_the_decided_level(tmp_path):
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.8), 24000)
    _write(tmp_path / "react_think_0.wav", _sine(0.02), 24000)
    gen.level_folder(tmp_path)
    for name in ("react_tool_0.wav", "react_think_0.wav"):
        x, _ = _read(tmp_path / name)
        assert abs(gen.active_db(x) - gen.VOICE_LEVEL_DB) < 0.1, name
    assert gen.VOICE_LEVEL_DB == -26.1


def test_the_placed_voices_are_at_the_decided_level():
    gen = _script()
    for voice in ("ja-jp-commercial-3", "ja-jp-csagent-8"):
        folder = gen.rc.TOOL_WAVS[0].parent / "voices" / voice
        for name in ("react_tool_0.wav", "react_think_0.wav"):
            x, params = gen._read_wav(folder / name)
            assert abs(gen.active_db(x, params.framerate) - gen.VOICE_LEVEL_DB) < 0.1, (voice, name)
