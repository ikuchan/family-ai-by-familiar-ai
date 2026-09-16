"""マイクの入力ゲイン（`AUDIO_INPUT_GAIN`・実機 2026-09-16 09:36）。

普通の声で話すと VAD が区間を 1.3 秒で短く切り、whisper は無音時の定型（「ご視聴ありがとう
ございました」）を出して `no_speech_prob=0.727` で捨てられた。大きめの声なら通る。マイク
（Yamaha YVC-300）のハード音量は上限（20/20・0 dB）で、これ以上は上げられない。

取り込んだ PCM をソフトで増幅する。VAD と whisper はどちらもその後段なので、両方に効く。
倍率は `.env` の `AUDIO_INPUT_GAIN`（既定 1.0＝いまと同じ）。int16 の範囲を超えた分は飽和
（クリップ）させる——折り返して逆相の雑音にしない。
"""

from __future__ import annotations

import numpy as np

from familiar_agent.tools.mic import apply_gain, input_gain_from_env


def _pcm(*values: int) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def test_gain_scales_the_samples():
    assert apply_gain(_pcm(100, -200, 0), 2.0) == _pcm(200, -400, 0)


def test_gain_of_one_returns_the_bytes_untouched():
    pcm = _pcm(123, -456)
    assert apply_gain(pcm, 1.0) is pcm


def test_gain_saturates_instead_of_wrapping():
    out = np.frombuffer(apply_gain(_pcm(30000, -30000), 2.0), dtype=np.int16)
    assert list(out) == [32767, -32768]


def test_empty_input_stays_empty():
    assert apply_gain(b"", 3.0) == b""


def test_the_env_value_is_read_as_a_multiplier(monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_GAIN", "2.5")
    assert input_gain_from_env() == 2.5


def test_missing_or_broken_env_means_no_gain(monkeypatch):
    monkeypatch.delenv("AUDIO_INPUT_GAIN", raising=False)
    assert input_gain_from_env() == 1.0
    monkeypatch.setenv("AUDIO_INPUT_GAIN", "loud")
    assert input_gain_from_env() == 1.0
    monkeypatch.setenv("AUDIO_INPUT_GAIN", "0")
    assert input_gain_from_env() == 1.0
