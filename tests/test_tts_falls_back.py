"""主が話せなかったら控えで話す（環-v・2026-09-24）。

主の担い手（Gemini）は外の API なので、**ネットが切れれば黙る**。家に居る相手から見れば、
それは壊れたのと区別がつかない。ローカルの SBV2 は読み 9/10・感情 △△△ と主に劣るが、
ネットが要らない。そこだけは代えられないので、控えに回す。

**控えは起動時から温めておく**（本人の決定・2026-09-24）。切れてからモデルを読むと 26.3 秒
黙ることになり、控えの役を果たさない。GPU 1.4 GB を常に占める代償は払う（`根拠台帳` §8 で
空きは 8.1 GB）。

**落ちるのは 1 回だけ。** 控えでも駄目なら、主が何と言ったかをそのまま返す——「何が起きたか」
が消えると、後から追えない。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

from familiar_agent.config import TTSConfig
from familiar_agent.tools.tts import TTSTool, ensure_sbv2_server

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


def _tool(*, fallback: str = "sbv2") -> TTSTool:
    return TTSTool(
        api_key="k",
        voice_id="v1",
        engine="gemini",
        gemini_api_key="g",
        fallback_engine=fallback,
    )


def _run(tool: TTSTool, *, gemini_ok: bool, sbv2_ok: bool):
    """Gemini と SBV2 の成否を決めて 1 回話させ、（返り, 通った担い手）を返す。"""
    used: list[str] = []

    async def _fake_play(path, gain=1.0):
        return True

    def _fake_gemini(_key):
        used.append("gemini")
        client = MagicMock()
        if not gemini_ok:
            client.models.generate_content_stream.side_effect = OSError("ネットが無い")
        else:
            part = MagicMock()
            part.inline_data.data = b"\x00\x01" * 1200
            part.inline_data.mime_type = "audio/pcm;rate=24000"
            chunk = MagicMock()
            chunk.candidates = [MagicMock(content=MagicMock(parts=[part]))]
            client.models.generate_content_stream.return_value = iter([chunk])
        return client

    async def _fake_sbv2(_self, _text):
        used.append("sbv2")
        if not sbv2_ok:
            raise OSError("合成サーバーが居ない")
        return _WAV

    with (
        patch("familiar_agent.tools.tts._play_local", new=_fake_play),
        patch("familiar_agent.tools.tts._gemini_client", new=_fake_gemini),
        patch.object(TTSTool, "_synth_sbv2", new=_fake_sbv2),
    ):
        result = asyncio.run(tool.say("こんにちは"))
    return result, used


# ── 落ちる ────────────────────────────────────────────────────────────────


def test_the_spare_speaks_when_the_main_one_cannot():
    result, used = _run(_tool(), gemini_ok=False, sbv2_ok=True)
    assert result.startswith("Said:")
    assert used == ["gemini", "sbv2"]


def test_the_spare_is_not_used_when_the_main_one_works():
    """**主が話せたら控えは呼ばない。** 同じ言葉を 2 度鳴らさない。"""
    result, used = _run(_tool(), gemini_ok=True, sbv2_ok=True)
    assert result.startswith("Said:")
    assert used == ["gemini"]


def test_both_failing_does_not_raise():
    """機器は落ちる前提のもの。声が出せないことでターンごと壊さない。"""
    result, used = _run(_tool(), gemini_ok=False, sbv2_ok=False)
    assert "話せなかった" in result
    assert used == ["gemini", "sbv2"]


def test_nothing_falls_back_when_no_spare_is_set():
    result, used = _run(_tool(fallback=""), gemini_ok=False, sbv2_ok=True)
    assert "話せなかった" in result
    assert used == ["gemini"]


def test_the_spare_is_not_the_main_one_again():
    """控えが主と同じなら、同じ失敗を 2 度やるだけ。呼ばない。"""
    tool = TTSTool(
        api_key="k", voice_id="v1", engine="gemini", gemini_api_key="g", fallback_engine="gemini"
    )
    result, used = _run(tool, gemini_ok=False, sbv2_ok=True)
    assert "話せなかった" in result
    assert used == ["gemini"]


# ── 控えを温めておく ──────────────────────────────────────────────────────


def _spawned(engine: str, fallback: str, output: str = "local") -> bool:
    with patch.dict(os.environ, {"TTS_ENGINE": engine, "TTS_FALLBACK": fallback}, clear=True):
        cfg = TTSConfig()
    with (
        patch("familiar_agent.tools.tts._sbv2_is_alive", return_value=False),
        patch("familiar_agent.tools.tts._spawn_sbv2") as spawn,
    ):
        ensure_sbv2_server(cfg, engine=cfg.engine, output=output)
    return spawn.called


def test_the_spare_is_warmed_up_at_startup():
    """主が Gemini でも、控えが SBV2 なら起こしておく。**これが今回足した道である。**"""
    assert _spawned("gemini", "sbv2") is True


def test_the_main_one_is_still_warmed_up():
    assert _spawned("sbv2", "") is True


def test_nothing_is_warmed_up_without_a_reason():
    """どこにも SBV2 を使わない構成では起こさない。GPU と十数秒を無駄にしない。"""
    assert _spawned("gemini", "") is False
    assert _spawned("elevenlabs", "") is False


def test_nothing_is_warmed_up_when_silent():
    """音を出さない構成（実機テスト用）でも起こさない。"""
    assert _spawned("gemini", "sbv2", output="silent") is False
