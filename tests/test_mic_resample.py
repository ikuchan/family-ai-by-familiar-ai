"""マイクの再標本化（48kHz → 16kHz）。

実機で常時集音の書き起こしが**話した内容と全く違うもの**になった。原因は再標本化である。

マイク（Yamaha YVC-300）の素の標本化率は **48,000 Hz** で、16,000 Hz へ落としている。
48,000 → 16,000 は正確に 3:1 なので、`np.interp` で位置を拾う実装は **3 サンプルおきに
間引くだけ**になっていた。**低域通過フィルタを通していない**ので、8 kHz より上の成分が
折り返して band 内の雑音になる（エイリアシング）。

録音ファイルで試したときに正しく出たのは、そのファイルを `arecord` で **16 kHz で直接**
録っており、再標本化を通らなかったからである。

したがって、間引く前に帯域を切る。`soxr.ResampleStream` を使う。**ブロックごとに呼ばれる**
ので、フィルタの状態をまたいで保つ必要があり（さもないと 96 ミリ秒ごとに継ぎ目の音が入る）、
`soxr` はその逐次の口を持っている。

実測（0.5 秒の正弦波・元の std は 14,142）:

| 入力 | 出力の std |
|---|---|
| 1,000 Hz | 14,141（そのまま通る） |
| 9,000 Hz | 88 |
| 12,000 Hz | 45 |
"""

from __future__ import annotations

import numpy as np

from familiar_agent.tools.mic import TARGET_RATE, _Resampler

_NATIVE = 48000


def _tone(freq: float, seconds: float, rate: int) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16).tobytes()


def _dominant_freq(pcm: bytes, rate: int) -> float:
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(arr * np.hanning(len(arr))))
    return float(np.fft.rfftfreq(len(arr), 1 / rate)[int(np.argmax(spectrum))])


def _level(pcm: bytes) -> float:
    """音の大きさ（std）。

    **割合では測れない。** 全部が小さくなっても、残りかすが「一番大きい成分」になるので
    比を取ると 0.78 のような値が出る。元の音の大きさと比べる。
    """
    return float(np.frombuffer(pcm, dtype=np.int16).astype(np.float64).std())


# ── 通す帯域 ───────────────────────────────────────────────────────────────

def test_a_voice_band_tone_survives_the_resampling():
    """1 kHz（声の帯域）はそのまま通る。周波数も大きさも保つ。"""
    original = _tone(1000.0, 0.5, _NATIVE)
    out = _Resampler(_NATIVE).process(original)
    assert abs(_dominant_freq(out, TARGET_RATE) - 1000.0) < 30.0
    assert _level(out) > _level(original) * 0.95


def test_the_output_length_matches_the_rate_ratio():
    """48kHz の 0.5 秒 → 16kHz のおよそ 0.5 秒（3 分の1 の標本数）。

    逐次の再標本化なので、フィルタの遅れぶん（実測 300 標本＝19 ミリ秒）が状態に残る。
    流し続ける経路なので、その遅れは次のブロックで出てくる。
    """
    out = _Resampler(_NATIVE).process(_tone(1000.0, 0.5, _NATIVE))
    assert 0 < int(TARGET_RATE * 0.5) - len(out) // 2 < 400


# ── 折り返しを止める（これが実機で壊れていた点）─────────────────────────────

def test_a_tone_above_nyquist_does_not_fold_back_into_the_band():
    """12 kHz は 16kHz では表せない。**4 kHz に化けてはいけない。**

    フィルタ無しの間引きでは、12 kHz がちょうど 4 kHz へ折り返して**元の大きさで**現れる。
    それが書き起こしを壊していた。
    """
    original = _tone(12000.0, 0.5, _NATIVE)
    out = _Resampler(_NATIVE).process(original)
    # フィルタ無しの間引きでは、ここが元と同じ大きさ（std 約 14,142）で 4 kHz に出る。
    assert _level(out) < _level(original) * 0.01, (
        f"12 kHz が残っている（出力 {_level(out):.0f}・元 {_level(original):.0f}）"
    )


def test_a_tone_just_above_the_cutoff_is_attenuated():
    """9 kHz も通してはいけない（16kHz の上限は 8 kHz）。"""
    original = _tone(9000.0, 0.5, _NATIVE)
    out = _Resampler(_NATIVE).process(original)
    assert _level(out) < _level(original) * 0.02


# ── ブロックをまたぐ ────────────────────────────────────────────────────────

def test_processing_in_blocks_matches_processing_all_at_once():
    """96 ミリ秒ずつ渡しても、まとめて渡したのと同じ音になる（継ぎ目を作らない）。

    フィルタの履歴と標本位置の端数をまたいで保っていなければ、ブロックの境目ごとに
    段差が入る。
    """
    pcm = _tone(1000.0, 0.5, _NATIVE)
    whole = _Resampler(_NATIVE).process(pcm)

    streamed = bytearray()
    engine = _Resampler(_NATIVE)
    step = int(_NATIVE * 0.096) * 2          # 96 ミリ秒ぶんのバイト数
    for i in range(0, len(pcm), step):
        streamed.extend(engine.process(pcm[i:i + step]))

    a = np.frombuffer(whole, dtype=np.int16).astype(np.float64)
    b = np.frombuffer(bytes(streamed), dtype=np.int16).astype(np.float64)
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) <= 2
    # 継ぎ目の段差があれば、差の大きさが元の音に対して無視できなくなる。
    assert np.abs(a[:n] - b[:n]).max() < a.std() * 0.05


def test_the_block_boundary_does_not_add_clicks():
    """境目に段差が無いことを、隣り合う標本の差の最大値で見る。

    1 kHz の正弦波なら、隣り合う標本の差は滑らかに収まる。継ぎ目があるとそこだけ跳ねる。
    """
    pcm = _tone(1000.0, 0.5, _NATIVE)
    engine = _Resampler(_NATIVE)
    streamed = bytearray()
    step = int(_NATIVE * 0.096) * 2
    for i in range(0, len(pcm), step):
        streamed.extend(engine.process(pcm[i:i + step]))

    arr = np.frombuffer(bytes(streamed), dtype=np.int16).astype(np.float64)
    jumps = np.abs(np.diff(arr))
    # 1 kHz を 16kHz で標本化した正弦波の隣接差は、振幅の 40% 程度が上限。
    assert jumps.max() < arr.std() * 1.5


# ── 落とさない経路 ─────────────────────────────────────────────────────────

def test_no_work_is_done_when_the_rate_already_matches():
    """16kHz のマイクなら何もしない（余計なフィルタを通さない）。"""
    pcm = _tone(1000.0, 0.1, TARGET_RATE)
    assert _Resampler(TARGET_RATE).process(pcm) == pcm


def test_an_empty_block_is_harmless():
    assert _Resampler(_NATIVE).process(b"") == b""
