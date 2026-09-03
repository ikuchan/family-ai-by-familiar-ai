"""STT のローカル化（出-a・faster-whisper）。

設計（用語一覧・`設計図_Mermaid` [D-知覚]）は STT＝faster-whisper を定めており、依存も
既に入っていた（未使用だった）。TTS のローカル化に続き、**聞き取りでも ElevenLabs の
無料枠を使わなくなる**。

**モデルは起動時に読む。** 読み込みに数秒かかるので、最初の書き起こしを待たせない。
使わない構成（`STT_ENGINE=elevenlabs`）では読まない。

**常時集音（`realtime_stt.py`）は次段。** あちらは自前の VAD（silero-vad）で区間を切って
逐次書き起こす作りへ変えることになり、規模が一段大きい。ここは「録音して起こす」経路だけ。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch


from familiar_agent.config import STTConfig
from familiar_agent.tools.stt import STTTool

_AUDIO = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 64


def _tool(engine: str = "whisper") -> STTTool:
    return STTTool(api_key="dummy-key", language="ja", engine=engine)


# ── Config ────────────────────────────────────────────────────────────────

def test_the_engine_defaults_to_whisper():
    with patch.dict(os.environ, {}, clear=True):
        assert STTConfig().engine == "whisper"


def test_the_engine_can_be_switched_back_to_elevenlabs():
    with patch.dict(os.environ, {"STT_ENGINE": "elevenlabs"}, clear=True):
        assert STTConfig().engine == "elevenlabs"


def test_the_model_and_quantisation_have_defaults():
    """VRAM の実測（本体 2.6GB＋SBV2 1.4GB＝4.2GB／空き 8.1GB）から int8_float16 を選んだ。"""
    with patch.dict(os.environ, {}, clear=True):
        cfg = STTConfig()
        assert cfg.whisper_model == "large-v3"
        assert cfg.whisper_compute_type == "int8_float16"


# ── 書き起こしの経路 ────────────────────────────────────────────────────────

def test_whisper_transcribes_locally_and_never_calls_elevenlabs():
    """ローカル化の要点。外部 API を叩かない（無料枠を使わない）。"""
    tool = _tool(engine="whisper")
    with patch.object(tool, "_transcribe_whisper", new=AsyncMock(return_value="おはよう")) as w, \
         patch.object(tool, "_transcribe_elevenlabs", new=AsyncMock(return_value="x")) as e, \
         patch.object(tool, "_record_mic", new=MagicMock(return_value=_AUDIO)):
        text = asyncio.run(tool.record_and_transcribe(asyncio.Event()))
    assert text == "おはよう"
    w.assert_awaited_once()
    e.assert_not_awaited()


def test_elevenlabs_is_still_reachable_when_selected():
    tool = _tool(engine="elevenlabs")
    with patch.object(tool, "_transcribe_whisper", new=AsyncMock(return_value="x")) as w, \
         patch.object(tool, "_transcribe_elevenlabs", new=AsyncMock(return_value="おはよう")) as e, \
         patch.object(tool, "_record_mic", new=MagicMock(return_value=_AUDIO)):
        text = asyncio.run(tool.record_and_transcribe(asyncio.Event()))
    assert text == "おはよう"
    w.assert_not_awaited()
    e.assert_awaited_once()


def test_empty_audio_never_reaches_the_model():
    """録れていなければモデルを呼ばない（無駄な GPU の呼び出しを避ける）。"""
    tool = _tool(engine="whisper")
    with patch.object(tool, "_transcribe_whisper", new=AsyncMock(return_value="x")) as w, \
         patch.object(tool, "_record_mic", new=MagicMock(return_value=None)):
        text = asyncio.run(tool.record_and_transcribe(asyncio.Event()))
    assert text == ""
    w.assert_not_awaited()


# ── モデルの読み込み ────────────────────────────────────────────────────────

def test_the_model_is_loaded_once_and_reused():
    """読み込みは数秒かかる。2回目以降は読み直さない。"""
    from familiar_agent.tools import stt

    # 読み込みの状態はモデル資源（MR）が持つので、リセットは入れ物ごと捨てる（出-c）。
    stt._whisper = None
    fake = MagicMock()
    with patch("familiar_agent.tools.stt._build_whisper_model", return_value=fake) as build:
        first = stt.load_whisper_model(STTConfig())
        second = stt.load_whisper_model(STTConfig())
    assert first is fake and second is fake
    build.assert_called_once()


def test_the_model_is_not_loaded_for_another_engine():
    """使わない構成では読まない（VRAM 2.5GB を無駄に占めない）。"""
    from familiar_agent.tools import stt

    # 読み込みの状態はモデル資源（MR）が持つので、リセットは入れ物ごと捨てる（出-c）。
    stt._whisper = None
    with patch.dict(os.environ, {"STT_ENGINE": "elevenlabs"}, clear=True), \
         patch("familiar_agent.tools.stt._build_whisper_model") as build:
        stt.ensure_whisper_model(STTConfig())
    build.assert_not_called()


def test_a_failed_load_does_not_raise():
    """モデルを読めなくても落とさない。書き起こしができないだけにする。"""
    from familiar_agent.tools import stt

    # 読み込みの状態はモデル資源（MR）が持つので、リセットは入れ物ごと捨てる（出-c）。
    stt._whisper = None
    with patch("familiar_agent.tools.stt._build_whisper_model", side_effect=OSError("no gpu")):
        assert stt.load_whisper_model(STTConfig()) is None


def test_a_failed_transcription_returns_empty_text():
    tool = _tool(engine="whisper")
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=None):
        text = asyncio.run(tool._transcribe_whisper(_AUDIO))
    assert text == ""
