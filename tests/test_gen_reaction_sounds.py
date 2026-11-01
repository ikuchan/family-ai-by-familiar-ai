"""作り置きの声を、指定した言葉だけ作る（`scripts/gen_reaction_sounds.py --only`・出-ce・2026-10-14）。

全部を作り直すと、聞いて決めた「えっと」の声まで変わる。作り直したい言葉だけを作る。「はーい！」の大きさは「えっと」に
揃える（出-df・2026-11-01・本人の決定ア）。
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
    got = [text for text, _ in gen._pairs({"はーい！"})]
    assert got == ["はーい！"]


def test_without_only_every_word_is_made():
    """**反証の側**：指定が無ければ、今までどおり全部。"""
    gen = _script()
    got = [text for text, _ in gen._pairs(None)]
    assert got == ["はーい！", "えっと"] and len(got) == len(gen.rc.TOOL_WORDS) + len(
        gen.rc.THINK_WORDS
    )


# ── 「はーい！」の大きさを「えっと」に揃える（出-df・2026-11-01・本人の決定ア） ─────────────

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


def test_a_folder_is_leveled_to_its_eto(tmp_path):
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.8), 24000)
    _write(tmp_path / "react_think_0.wav", _sine(0.1), 24000)
    eto_before = (tmp_path / "react_think_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    hai, fmt = _read(tmp_path / "react_tool_0.wav")
    eto, _ = _read(tmp_path / "react_think_0.wav")
    assert abs(gen.active_db(hai) - gen.active_db(eto)) < 0.2
    assert fmt == (1, 2, 24000)
    assert (tmp_path / "react_think_0.wav").read_bytes() == eto_before  # 反証の側：えっとは変えない


def test_a_silent_file_is_left_untouched(tmp_path):
    """無音の「えっと」に合わせて「はーい！」を消さない。"""
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.8), 24000)
    _write(tmp_path / "react_think_0.wav", np.zeros(12000), 24000)
    before = (tmp_path / "react_tool_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    assert (tmp_path / "react_tool_0.wav").read_bytes() == before


def test_a_matched_file_is_not_rewritten(tmp_path):
    """もう揃っていれば書き直さない（作り直すたびに丸めの誤差を積もらせない）。"""
    gen = _script()
    _write(tmp_path / "react_tool_0.wav", _sine(0.3), 24000)
    _write(tmp_path / "react_think_0.wav", _sine(0.3), 24000)
    before = (tmp_path / "react_tool_0.wav").read_bytes()
    gen.level_folder(tmp_path)
    assert (tmp_path / "react_tool_0.wav").read_bytes() == before


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
    hai, _ = _read(tmp_path / "react_tool_0.wav")
    eto, _ = _read(tmp_path / "react_think_0.wav")
    assert abs(gen.active_db(hai, 44100) - gen.active_db(eto, 44100)) < 0.05
