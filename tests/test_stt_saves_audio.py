"""書き起こした区切りの音を残す（知-z-は・2026-10-07・調べるための道具）。

名前「パジュ」が書き起こしから落ちる（頭が欠けて「ジュ」になる・まるごと落ちる・実機 20:31〜20:34）が、書き起こした
音を残していないので聞き直せない。`STT_SAVE_AUDIO=1` のときだけ、区切りの音（16kHz・モノの WAV）と記録（時刻・文・
無音らしさ・確かさ・秒）を `STT_SAVE_AUDIO_DIR`（既定 `~/.cache/familiar-ai/stt_audio`）に残す。最新 20 件まで。書き起こしは
止めない。文が空（幻聴として捨てた・名前が落ちた）でも残す。
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import soundfile as sf

from familiar_agent.config import STTConfig
from tests.test_local_stt import _engine

_AUDIO = (np.arange(16000, dtype=np.int16) % 200).tobytes()  # 1 秒


def _cfg(tmp_path, on=True, keep=20):
    cfg = STTConfig()
    cfg.save_audio = on
    cfg.save_audio_dir = str(tmp_path)
    cfg.save_audio_max = keep
    return cfg


def _write(engine, audio=_AUDIO):
    engine._last_measures = (0.12, -0.5)

    def transcribe(a):
        engine._last_measures = (0.12, -0.5)
        return "ジュ音楽をかけて"

    engine._transcribe = transcribe
    asyncio.run(engine._write(audio))


def test_nothing_is_saved_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("STT_SAVE_AUDIO", raising=False)
    assert STTConfig().save_audio is False
    engine = _engine(vad_says=[], cfg=_cfg(tmp_path, on=False))
    _write(engine)
    assert list(tmp_path.iterdir()) == []


def test_the_segment_and_its_record_are_saved(tmp_path):
    engine = _engine(vad_says=[], cfg=_cfg(tmp_path))
    _write(engine)
    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    data, rate = sf.read(str(wavs[0]), dtype="int16")
    assert rate == 16000 and data.tobytes() == _AUDIO
    rec = json.loads(wavs[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert rec["text"] == "ジュ音楽をかけて"
    assert (rec["no_speech"], rec["logprob"]) == (0.12, -0.5)
    assert rec["seconds"] == 1.0


def test_an_empty_transcript_is_saved_too(tmp_path):
    engine = _engine(vad_says=[], cfg=_cfg(tmp_path))
    engine._transcribe = lambda a: ""
    asyncio.run(engine._write(_AUDIO))
    (j,) = tmp_path.glob("*.json")
    assert json.loads(j.read_text(encoding="utf-8"))["text"] == ""


def test_only_the_latest_are_kept(tmp_path):
    engine = _engine(vad_says=[], cfg=_cfg(tmp_path, keep=3))
    for _ in range(5):
        _write(engine)
    assert len(list(tmp_path.glob("*.wav"))) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_a_save_failure_does_not_stop_the_transcript(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")  # 保存先がファイル（ディレクトリを作れない）
    engine = _engine(vad_says=[], cfg=_cfg(blocker))
    _write(engine)
    assert engine.on_committed.get_nowait() == "ジュ音楽をかけて"
    assert "残せなかった" in caplog.text
