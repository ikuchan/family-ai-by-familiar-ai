"""常時集音のローカル化（出-a・silero-vad ＋ faster-whisper）。

ElevenLabs の WebSocket はサーバー側で VAD を持ち、区間を切って書き起こしを返していた。
ローカル化すると**その両方を自分でやる**ことになる。

- 区間を切る … `silero_vad.VADIterator`（逐次・16kHz・**512 サンプル固定**）
- 書き起こす … `faster-whisper`（`stt.py` のモデルを共有）

**無音の窓は VAD が内部で持つ**（`min_silence_duration_ms`・既定 1.0 秒＝ElevenLabs と同じ）。
`VADIterator` は**境目でしか値を返さない**（`start`／`end`。発話中も無音中も `None`）ので、
こちらでフレームを数えて無音を測ることはできない。実装当初それを誤り、発話が始まった直後から
無音を数えて 1 秒で切っていた（実測で崩れた原因はこれだった）。

**部分結果は出さない。** 途中で何度も書き起こすと GPU を無駄に回す（発話 3 秒なら 6 回）。
代わりに発話の始まりを知らせ、GUI が「聞いています」の印を出す。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from familiar_agent.config import STTConfig
from familiar_agent.tools.local_stt import FRAME_SAMPLES, LocalSttEngine

_RATE = 16000
# 16bit PCM。無音は 0、発話は振幅のある波形（VAD は差し替えるので中身は問わない）。
_SILENT_FRAME = b"\x00\x00" * FRAME_SAMPLES
_LOUD_FRAME = b"\x00\x40" * FRAME_SAMPLES


def _engine(*, vad_says, transcribe="おはよう", cfg=None):
    """VAD の判定を差し替えたエンジンを作る。

    `vad_says` は 1 フレームごとに VAD が返す値の列。**境目だけに値が入る**
    （`None`＝境目なし・`"start"`＝発話開始・`"end"`＝発話終了）。
    """
    cfg = cfg or STTConfig()
    engine = LocalSttEngine(cfg)
    calls = iter(vad_says)
    engine._vad_step = MagicMock(side_effect=lambda frame: next(calls, None))
    engine._transcribe = MagicMock(return_value=transcribe)
    engine.on_committed = asyncio.Queue()
    return engine


def _feed(engine, frames: list[bytes]) -> None:
    async def run():
        for f in frames:
            await engine.feed(f)
    asyncio.run(run())


# ── フレームの単位 ─────────────────────────────────────────────────────────

def test_the_frame_size_is_what_silero_requires():
    """silero-vad は 16kHz で 512 サンプル固定。ここを間違えると判定が壊れる。"""
    assert FRAME_SAMPLES == 512


def test_the_mic_block_is_a_multiple_of_the_frame():
    """マイクのブロックを 512 の倍数にしておく（緩衝の余りを持ち越さない）。"""
    from familiar_agent.tools.mic import TARGET_RATE, _BLOCK_MS

    samples = TARGET_RATE * _BLOCK_MS // 1000
    assert samples % FRAME_SAMPLES == 0, f"{samples} は {FRAME_SAMPLES} の倍数でない"


# ── 区間の切り方 ───────────────────────────────────────────────────────────

def test_silence_alone_never_reaches_the_model():
    """黙っているあいだは書き起こさない（GPU を無駄に回さない）。"""
    engine = _engine(vad_says=[None] * 10)
    _feed(engine, [_SILENT_FRAME] * 10)
    engine._transcribe.assert_not_called()
    assert engine.on_committed.empty()


def test_a_segment_is_committed_after_the_silence_window():
    """発話 → 無音が続いたら区間を確定し、書き起こす。

    長さは `min_segment_sec` 以上にする（短い区間は持ち越される＝別のテストで見る）。
    """
    long_frames = int(1.6 * _RATE / FRAME_SAMPLES)
    engine = _engine(vad_says=["start"] + [None] * (long_frames - 1) + ["end"])
    _feed(engine, [_LOUD_FRAME] * (long_frames + 1))
    engine._transcribe.assert_called_once()
    assert engine.on_committed.get_nowait() == "おはよう"


def test_a_short_pause_does_not_end_the_segment():
    """息継ぎで切らない。VAD が `end` を返さないあいだは確定しない。"""
    engine = _engine(vad_says=["start"] + [None] * 8)
    _feed(engine, [_LOUD_FRAME] * 3 + [_SILENT_FRAME] * 6)
    engine._transcribe.assert_not_called()


def test_a_very_long_segment_is_cut_at_the_limit():
    """長すぎる区間は区切る。雑音が続いたときにメモリと GPU を食い続けないため。"""
    cfg = STTConfig()
    frames = int(cfg.max_segment_sec * _RATE / FRAME_SAMPLES) + 2
    engine = _engine(vad_says=["start"] + [None] * frames)
    _feed(engine, [_LOUD_FRAME] * frames)
    engine._transcribe.assert_called_once()


# ── 発話の始まりを知らせる ──────────────────────────────────────────────────

def test_the_start_of_speech_is_announced_once():
    """GUI が「聞いています」を出すための合図。始まりで1回だけ。"""
    seen: list[str] = []
    engine = _engine(vad_says=["start"] + [None] * 5)
    engine.on_speech_start = lambda: seen.append("start")
    _feed(engine, [_LOUD_FRAME] * 6)
    assert seen == ["start"]


# ── 失敗しても落とさない ────────────────────────────────────────────────────

def test_a_failed_transcription_does_not_raise():
    long_frames = int(1.6 * _RATE / FRAME_SAMPLES)
    engine = _engine(vad_says=["start"] + [None] * (long_frames - 1) + ["end"])
    engine._transcribe = MagicMock(side_effect=OSError("gpu が無い"))
    _feed(engine, [_LOUD_FRAME] * (long_frames + 1))
    assert engine.on_committed.empty()      # 何も配らないが、例外も出さない


# ── どちらの担い手を使うか ──────────────────────────────────────────────────

def test_the_local_engine_is_not_used_for_elevenlabs():
    from familiar_agent.tools.local_stt import should_use_local

    with patch.dict(os.environ, {"STT_ENGINE": "elevenlabs"}, clear=True):
        assert should_use_local(STTConfig()) is False
    with patch.dict(os.environ, {}, clear=True):
        assert should_use_local(STTConfig()) is True


def test_the_silence_window_and_limit_have_defaults():
    with patch.dict(os.environ, {}, clear=True):
        cfg = STTConfig()
        assert cfg.vad_silence_sec == pytest.approx(1.0)
        assert cfg.max_segment_sec == pytest.approx(30.0)


# ── 短い区間を繋ぐ ─────────────────────────────────────────────────────────
#
# 実測（実機の録音）で、無音 1.0 秒の窓だと「今日は／7月30日10時45分／天気は曇りです」が
# 3つに分断され、真ん中の 1.0 秒の区間が 'ジュージュージュー' に崩れた。一括で起こすと
# 正しかった（'今日は7月30日10時45分天気は曇りです'）。**短い断片では文脈が足りない。**
#
# そこで、短い区間は確定させず次の発話まで持ち越して合わせる。ただし次が来ないまま
# 無音が続いたら諦めて単独で起こす（「はい」だけの返事が永久に届かないのを避ける）。

def test_a_short_segment_waits_for_the_next_utterance():
    """1.5 秒未満の区間は、その場では書き起こさない。"""
    # 発話 3 フレーム（約 0.1 秒）→ `end` で区間の終わり
    engine = _engine(vad_says=["start", None, "end"])
    _feed(engine, [_LOUD_FRAME] * 3)
    engine._transcribe.assert_not_called()
    assert engine.on_committed.empty()


def test_a_short_segment_is_merged_into_the_next_one():
    """次の発話が来たら合わせて1つにする（文脈が繋がる）。"""
    long_frames = int(1.6 * _RATE / FRAME_SAMPLES)      # 1.6 秒＝しきい値より長い
    engine = _engine(
        vad_says=(["start", None, "end"]                        # 短い発話
                  + ["start"] + [None] * (long_frames - 1) + ["end"]),   # 長い発話
    )
    _feed(engine, [_LOUD_FRAME] * 3 + [_LOUD_FRAME] * (long_frames + 1))
    engine._transcribe.assert_called_once()
    # 渡された音声に、短い区間の分も含まれている（合わせて1つ）。
    passed = engine._transcribe.call_args.args[0]
    assert len(passed) > long_frames * FRAME_SAMPLES * 2


def test_a_long_segment_is_committed_on_its_own():
    """1.5 秒以上なら、そのまま書き起こす（持ち越さない）。"""
    long_frames = int(1.6 * _RATE / FRAME_SAMPLES)
    engine = _engine(vad_says=["start"] + [None] * (long_frames - 1) + ["end"])
    _feed(engine, [_LOUD_FRAME] * (long_frames + 1))
    engine._transcribe.assert_called_once()


def test_a_held_segment_is_eventually_given_up_and_committed():
    """次が来なくても、時間が経てば諦めて配る（短い返事を失わない）。

    諦める時機は**時刻**で測る（VAD は境目しか返さないので、無音のフレーム数は数えられない）。
    """
    engine = _engine(vad_says=["start", None, "end"] + [None] * 5)

    async def run():
        for f in [_LOUD_FRAME] * 3:
            await engine.feed(f)
        engine._transcribe.assert_not_called()      # まだ持ち越している
        engine._held_since = 0.0                    # 十分に時間が経った状態にする
        for f in [_SILENT_FRAME] * 2:
            await engine.feed(f)

    asyncio.run(run())
    engine._transcribe.assert_called_once()
    assert engine.on_committed.get_nowait() == "おはよう"


def test_the_merge_thresholds_have_defaults():
    with patch.dict(os.environ, {}, clear=True):
        cfg = STTConfig()
        assert cfg.min_segment_sec == pytest.approx(1.5)
        assert cfg.hold_give_up_sec == pytest.approx(3.0)


# ── モデルへ渡す形 ─────────────────────────────────────────────────────────
#
# 実機で書き起こしが全く通らなかった。WAV に書いて `BytesIO` を渡していたため、
# faster-whisper がそれを **PyAV** で復号し 16kHz へ再標本化していた。その再標本化が
# `EAGAIN` を返したとき、**PyAV が errno の説明文を ascii で読もうとして落ちる**
# （この環境は LANG=ja_JP.UTF-8 なので説明文が日本語）。
#
#   File "av/error.pyx", line 421, in av.error.err_check
#   UnicodeDecodeError: 'ascii' codec can't decode byte 0xe3 in position 0
#
# **そもそも復号も再標本化も要らない。** 音はすでに 16kHz の PCM である。float32 の
# numpy 配列をそのまま渡せば `decode_audio` を通らず、PyAV が経路から消える。

def test_the_audio_reaches_the_model_as_float32_samples():
    """WAV でもファイル様オブジェクトでもなく、16kHz の float32 配列を渡す。"""
    import numpy as np

    cfg = STTConfig()
    engine = LocalSttEngine(cfg)
    model = MagicMock()
    model.transcribe.return_value = ([], MagicMock(duration=1.0))

    pcm = (np.arange(_RATE, dtype=np.int16) % 1000).tobytes()   # 1 秒ぶん
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        engine._transcribe(pcm)

    passed = model.transcribe.call_args.args[0]
    assert isinstance(passed, np.ndarray), f"配列でなく {type(passed)} を渡している"
    assert passed.dtype == np.float32
    assert len(passed) == _RATE                 # 16kHz のまま（再標本化していない）
    assert abs(passed).max() <= 1.0             # -1.0〜1.0 に正規化してある


def test_the_model_is_never_asked_to_decode_a_container():
    """PyAV を経路から外したことの反証側。ファイル様オブジェクトを渡していたら落ちる。"""
    import io

    engine = LocalSttEngine(STTConfig())
    model = MagicMock()
    model.transcribe.return_value = ([], MagicMock(duration=1.0))
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        engine._transcribe(b"\x00\x01" * _RATE)

    passed = model.transcribe.call_args.args[0]
    assert not isinstance(passed, (io.IOBase, bytes, bytearray))


# ── 幻聴の切り分け（実機で「ご視聴ありがとうございました」が出た）─────────────
#
# 話していないのに定型句が書き起こされた。Whisper は音声でないもの（無音・物音・息）を
# 渡されると、学習データの字幕によくある文を当てはめる。silero-vad は音の有無しか見ない
# ので、物音が区間として切り出されると whisper へ渡って幻聴になる。
#
# 弾く前に**測る**。faster-whisper はセグメントごとに `no_speech_prob`（音声でない確率）と
# `avg_logprob`（平均対数尤度）を返すので、幻聴のときと実際に話したときの値を集めてから
# 閾値を決める。ここではまだ何も捨てない。
#
# 本文は info 以上へ出さない（会話内容のため）。数値と字数だけを info に出す。

def test_each_segment_reports_its_no_speech_probability(caplog):
    """セグメントごとに `no_speech_prob` と `avg_logprob` を残す。"""
    import logging

    import numpy as np

    seg = MagicMock(text="ご視聴ありがとうございました", no_speech_prob=0.87,
                    avg_logprob=-0.42, start=0.0, end=1.4)
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock(duration=1.4))

    engine = LocalSttEngine(STTConfig())
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        with caplog.at_level(logging.INFO, logger="familiar_agent.tools.local_stt"):
            engine._transcribe((np.zeros(_RATE, dtype=np.int16)).tobytes())

    line = " ".join(caplog.messages)
    assert "no_speech_prob=0.870" in line, f"確率を残していない: {line}"
    assert "avg_logprob=-0.420" in line


def test_the_measurement_line_does_not_leak_the_transcript(caplog):
    """本文は info へ出さない（会話内容のため）。字数だけにする。"""
    import logging

    import numpy as np

    seg = MagicMock(text="秘密の話", no_speech_prob=0.1, avg_logprob=-0.2,
                    start=0.0, end=1.0)
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock(duration=1.0))

    engine = LocalSttEngine(STTConfig())
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        with caplog.at_level(logging.INFO, logger="familiar_agent.tools.local_stt"):
            engine._transcribe((np.zeros(_RATE, dtype=np.int16)).tobytes())

    assert "秘密の話" not in " ".join(caplog.messages)


# ── 幻聴を捨てる（`no_speech_prob` で切る）─────────────────────────────────────
#
# 実機で15件を測り、ラベルを付けたところ、次のように完全に分かれた。
#
# | `no_speech_prob` | 種別 |
# |---|---|
# | 0.890 / 0.880 / 0.849 / 0.799 / 0.785 / 0.722 | 幻聴（「ご視聴ありがとうございました」等） |
# | 0.709 / 0.684 / 0.502 / 0.429 / 0.373 / 0.269 / 0.250 / 0.209 / 0.139 | 本物 |
#
# 幻聴は全件 0.722 以上、本物は全件 0.709 以下で、取り違えは無い。境目の直下 **0.72** を
# 既定にする。`avg_logprob` は幻聴 −0.352〜−0.746・本物 −0.138〜−0.712 で重なるので使わない。
#
# **`no_speech_prob` は 30 秒の窓ごとの値**で、同じ書き起こしのセグメントは同じ値を持つ。
# よってセグメント単位ではなく、その書き起こしをまるごと捨てる。

def test_a_hallucination_above_the_threshold_is_dropped():
    """0.72 を超えたら、書き起こしをまるごと捨てる（実測 0.785 の幻聴）。"""
    import numpy as np

    seg = MagicMock(text="ご視聴ありがとうございました", no_speech_prob=0.785,
                    avg_logprob=-0.580, start=0.0, end=30.0)
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock(duration=3.1))

    engine = LocalSttEngine(STTConfig())
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        assert engine._transcribe((np.zeros(_RATE, dtype=np.int16)).tobytes()) == ""


def test_a_quiet_real_utterance_just_below_the_threshold_survives():
    """0.709 は本物だった（「はい」のような短い返事）。落としてはいけない。"""
    import numpy as np

    seg = MagicMock(text="うん", no_speech_prob=0.709, avg_logprob=-0.712,
                    start=0.0, end=2.0)
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock(duration=1.86))

    engine = LocalSttEngine(STTConfig())
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        assert engine._transcribe((np.zeros(_RATE, dtype=np.int16)).tobytes()) == "うん"


def test_the_no_speech_ceiling_has_a_default_and_can_be_moved():
    """既定は実測の境目 0.72。実機で外れが出たら環境変数で動かす。"""
    import os
    from unittest.mock import patch as _patch

    with _patch.dict(os.environ, {}, clear=True):
        assert STTConfig().no_speech_max == pytest.approx(0.72)
    with _patch.dict(os.environ, {"STT_NO_SPEECH_MAX": "0.9"}, clear=True):
        assert STTConfig().no_speech_max == pytest.approx(0.9)
