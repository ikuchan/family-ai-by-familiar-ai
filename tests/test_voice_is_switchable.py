"""声を切り替えられるようにし、既定を Laomedeia にする（出-bv・2026-10-12・本人の決定）。

10/12 の聞き比べで、本人は Charon を「おっさん」と言い、Puck と Laomedeia のどちらかを使うことにした（`根拠台帳` §45）。
作り置きの声（合図・残高切れの知らせ）は SBV2 で作った 1 組だけで、返事（Gemini）と声が違って「気持ち悪い」。
声ごとのフォルダ `sounds/voices/<声>/` に作り置きの声を置き、`.env` の `GEMINI_TTS_VOICE` で全部がそろって切り替わる。
"""

from __future__ import annotations

import asyncio
import io
import os
import pathlib
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.config import TTSConfig
from familiar_agent.core import reaction_cue as rc

SOUNDS = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent" / "sounds"


def test_the_voice_defaults_to_laomedeia():
    with patch.dict(os.environ, {}, clear=True):
        assert TTSConfig().gemini_voice == "Laomedeia"


def test_the_voice_follows_the_env():
    """**反証の側**：`.env` で変えれば、その声になる。"""
    with patch.dict(os.environ, {"GEMINI_TTS_VOICE": "Puck"}, clear=True):
        assert TTSConfig().gemini_voice == "Puck"


# ── 作り置きの声は、いまの声のフォルダから ───────────────────────────────


def _tree(tmp_path, *, voiced: bool):
    base = tmp_path / "react_tool_0.wav"
    base.write_bytes(b"RIFF")
    if voiced:
        d = tmp_path / "voices" / "Puck"
        d.mkdir(parents=True)
        (d / "react_tool_0.wav").write_bytes(b"RIFF")
    return base


def test_the_voice_folder_is_used(tmp_path):
    base = _tree(tmp_path, voiced=True)
    with patch.dict(os.environ, {"TTS_ENGINE": "gemini", "GEMINI_TTS_VOICE": "Puck"}):
        assert rc.voiced(base) == tmp_path / "voices" / "Puck" / "react_tool_0.wav"


def test_the_base_is_used_without_the_voice_file(tmp_path):
    """**反証の側**：その声のファイルが無ければ、直下（SBV2 版）で鳴らす。"""
    base = _tree(tmp_path, voiced=False)
    with patch.dict(os.environ, {"TTS_ENGINE": "gemini", "GEMINI_TTS_VOICE": "Puck"}):
        assert rc.voiced(base) == base


def test_an_sbv2_main_voice_uses_the_sbv2_files(tmp_path):
    """主が SBV2 なら、声をそろえるため SBV2 版を使う。"""
    base = _tree(tmp_path, voiced=True)
    with patch.dict(os.environ, {"TTS_ENGINE": "sbv2", "GEMINI_TTS_VOICE": "Puck"}):
        assert rc.voiced(base) == base


def test_the_cue_comes_from_the_voice_folder():
    with patch.dict(os.environ, {"TTS_ENGINE": "gemini", "GEMINI_TTS_VOICE": "Laomedeia"}):
        assert rc.sound_for(rc.TOOL).parent == SOUNDS / "voices" / "Laomedeia"


def test_the_credit_voice_comes_from_the_voice_folder(monkeypatch, tmp_path):
    from familiar_agent.io import dif

    (tmp_path / "credit_gemini.wav").write_bytes(b"RIFF")
    d = tmp_path / "voices" / "Puck"
    d.mkdir(parents=True)
    (d / "credit_gemini.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(dif, "_SOUNDS", tmp_path)
    played = AsyncMock(return_value=True)
    monkeypatch.setattr(dif, "_play_wav", played)
    with patch.dict(os.environ, {"TTS_ENGINE": "gemini", "GEMINI_TTS_VOICE": "Puck"}):
        assert asyncio.run(dif.DIF(ip=MagicMock()).say_credit("gemini", gain=0.25)) is True
    assert played.await_args.args == (d / "credit_gemini.wav", 0.25)


# ── 作成スクリプトが使う、Gemini で 1 文を WAV にする口 ─────────────────────


def _client(chunks):
    c = MagicMock()
    c.models.generate_content_stream.return_value = chunks
    return c


def _chunk(data: bytes):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type="audio/pcm;rate=24000"))
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def test_gemini_wav_wraps_the_pcm(monkeypatch):
    from familiar_agent.tools import tts

    pcm = b"\x01\x00" * 2400  # 0.1 秒
    client = _client([_chunk(pcm)])
    monkeypatch.setattr(tts, "_gemini_client", lambda key: client)
    wav = tts.gemini_wav(TTSConfig(), "はい", "Puck")
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (24000, 1, 2400)
    assert "Puck" in str(client.models.generate_content_stream.call_args.kwargs["config"])


def test_gemini_wav_fails_without_sound(monkeypatch):
    """**反証の側**：音が返らなければ、空の WAV を書かずに例外にする。"""
    from familiar_agent.tools import tts

    monkeypatch.setattr(tts, "_gemini_client", lambda key: _client([]))
    with pytest.raises(RuntimeError):
        tts.gemini_wav(TTSConfig(), "はい", "Puck")


# ── 2 組そろっている ───────────────────────────────────────────────────


@pytest.mark.parametrize("voice", ["Puck", "Laomedeia"])
def test_each_voice_has_all_recordings(voice):
    from familiar_agent.core.credit import READINGS

    d = SOUNDS / "voices" / voice
    names = [p.name for p in (*rc.TOOL_WAVS, *rc.THINK_WAVS)] + [
        f"credit_{n}.wav" for n in READINGS
    ]
    for name in names:
        wav = d / name
        assert wav.exists() and wav.stat().st_size > 2_000, wav
        assert wav.read_bytes()[:4] == b"RIFF"
