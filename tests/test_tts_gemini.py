"""声の担い手を Gemini にする（環-v・2026-09-24）。

実機で詰まった 5 文を**漢字のまま** 12 とおりに読ませ、本人が ○△× を付けた（`根拠台帳` §45）。
いまの ElevenLabs flash ＋ ひらがな化は **5 文とも ×**、`gemini-3.8-flash-lite-tts` の声
`Charon` は **5 文とも ○** で、感情の出し分けも ○○○ だった。費用は月 $11.00 → $1.32。

**ひらがな化を通さない。** 漢字をそのまま読めることが、この担い手を選んだ理由である。
迂回（`core/reading.for_speech`）を残すと、直った読みをまた崩しかねない。

**角括弧タグも渡さない。** `[cheerful]` は `eleven_v3` だけが解する指示で、Gemini には
意味が無い。感情は文そのもので言い分ける。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.config import TTSConfig
from familiar_agent.tools.tts import TTSTool

#: Gemini の逐次の返りは**生 PCM**（24kHz・16bit・mono）。一括は WAV で返る。両方を扱う。
_PCM = b"\x00\x01" * 2400
_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


def _tool(engine: str = "gemini", output: str = "local") -> TTSTool:
    return TTSTool(
        api_key="dummy-key",
        voice_id="v1",
        output=output,
        engine=engine,
        gemini_api_key="dummy-gemini-key",
    )


def _chunk(data: bytes, mime: str):
    part = MagicMock()
    part.inline_data.data = data
    part.inline_data.mime_type = mime
    chunk = MagicMock()
    chunk.candidates = [MagicMock(content=MagicMock(parts=[part]))]
    return chunk


# ── Config ────────────────────────────────────────────────────────────────


def test_the_model_defaults_to_flash_lite():
    """読み 10/10・感情 ○○○ で、月 $1.32 と一番安かった組み合わせ。"""
    with patch.dict(os.environ, {}, clear=True):
        assert TTSConfig().gemini_model == "gemini-3.8-flash-lite-tts"


def test_the_voice_defaults_to_charon():
    """同じモデルでも声で点が変わった——Puck は感情 △△△、Charon は ○○○。"""
    with patch.dict(os.environ, {}, clear=True):
        assert TTSConfig().gemini_voice == "Charon"


def test_the_model_and_voice_can_be_changed():
    with patch.dict(
        os.environ,
        {"GEMINI_TTS_MODEL": "gemini-3.8-flash-tts", "GEMINI_TTS_VOICE": "Puck"},
        clear=True,
    ):
        cfg = TTSConfig()
        assert cfg.gemini_model == "gemini-3.8-flash-tts"
        assert cfg.gemini_voice == "Puck"


def test_the_key_falls_back_to_the_utility_one():
    """軽量LLM がすでに Gemini なので、鍵は同じものが使える（`UTILITY_API_KEY`）。"""
    with patch.dict(os.environ, {"UTILITY_API_KEY": "u-key"}, clear=True):
        assert TTSConfig().gemini_api_key == "u-key"


def test_a_dedicated_key_wins():
    with patch.dict(
        os.environ, {"UTILITY_API_KEY": "u-key", "GEMINI_TTS_API_KEY": "tts-key"}, clear=True
    ):
        assert TTSConfig().gemini_api_key == "tts-key"


# ── 渡す文を加工しない ────────────────────────────────────────────────────


def test_the_text_is_not_turned_into_hiragana():
    """漢字をそのまま読めるのが、この担い手を選んだ理由である。"""
    tool = _tool()
    assert tool._text_for_synth("出入口の押入に金木犀") == "出入口の押入に金木犀"


def test_elevenlabs_still_gets_hiragana():
    """迂回は ElevenLabs のときだけ残す（戻せるようにしてある）。"""
    tool = _tool(engine="elevenlabs")
    got = tool._text_for_synth("出入口")
    assert got != "出入口"


def test_bracket_tags_are_not_offered():
    """`[cheerful]` を解するのは `eleven_v3` だけ。Gemini へ渡せばそのまま音になる。"""
    assert _tool().understands_tags is False


# ── 合成して鳴らす ────────────────────────────────────────────────────────


def _say(tool: TTSTool, chunks: list, text: str = "こんにちは"):
    played: list = []

    async def _fake_play(path, gain=1.0):
        played.append(path)
        return True

    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(chunks)
    with (
        patch("familiar_agent.tools.tts._play_local", new=_fake_play),
        patch("familiar_agent.tools.tts._gemini_client", return_value=client),
    ):
        result = asyncio.run(tool.say(text))
    return result, played, client


def test_a_stream_of_raw_pcm_is_spoken():
    result, played, _ = _say(_tool(), [_chunk(_PCM, "audio/pcm;rate=24000")])
    assert result.startswith("Said:")
    assert played, "鳴らしていない"


def test_a_wav_chunk_is_spoken_too():
    """一括の返りは WAV。包み直さずそのまま渡す。"""
    result, played, _ = _say(_tool(), [_chunk(_WAV, "audio/wav")])
    assert result.startswith("Said:")
    assert played


def test_the_model_and_voice_reach_the_call():
    _, _, client = _say(_tool(), [_chunk(_PCM, "audio/pcm;rate=24000")])
    kwargs = client.models.generate_content_stream.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash-lite-tts"
    assert "Charon" in str(kwargs["config"])


def test_a_failure_does_not_raise():
    """機器は落ちる前提のもの。声が出せないことでターンごと壊さない。"""
    tool = _tool()
    client = MagicMock()
    client.models.generate_content_stream.side_effect = OSError("ネットが無い")
    with (
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("familiar_agent.tools.tts._gemini_client", return_value=client),
    ):
        result = asyncio.run(tool.say("こんにちは"))  # 控えは置いていない（`_tool()` の既定）
    assert "話せなかった" in result


def test_silent_does_not_synthesise():
    """実機テスト用。合成も走らせない（費用も待ちも生まない）。"""
    tool = _tool(output="silent")
    with patch("familiar_agent.tools.tts._gemini_client") as client:
        result = asyncio.run(tool.say("こんにちは"))
    assert result.startswith("Said (silent)")
    client.assert_not_called()


def test_the_voice_guard_is_told():
    """自分の声を自分で聞き返さないための門。担い手が変わっても同じように通す。"""
    guard = MagicMock()
    tool = TTSTool(
        api_key="k", voice_id="v1", engine="gemini", gemini_api_key="g", voice_guard=guard
    )
    _say(tool, [_chunk(_PCM, "audio/pcm;rate=24000")], text="ただいま")
    guard.on_tts_start.assert_called_once_with("ただいま")
    assert guard.on_tts_end.call_args.args[0] == "ただいま"
