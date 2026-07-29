"""TTS のローカル化（出-a・Style-Bert-VITS2）。

設計（`計測・設定値 根拠台帳` §9）は TTS＝Style-Bert-VITS2・声＝jvnv-M2-jp を確定と
している。実機の聴き比べも済んでおり、モデルと専用の仮想環境は `~/tts_eval/` にある。

**別プロセスの HTTP サーバーとして動かす。** 本体は Python 3.11・torch 2.10.0+cu128 だが、
SBV2 は Python 3.12・torch 2.5.1+cu121・numpy 1.26.4 固定で、同じプロセスには載らない。
AGPL-3.0 の結合を弱める意味もある。

**感情（PAD → style）の写像はまだ入れない。** style は固定（Neutral）で、写像は次段。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch


from familiar_agent.config import TTSConfig
from familiar_agent.tools.tts import TTSTool

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32   # 中身は問わない（再生は差し替える）


def _tool(engine: str = "sbv2", output: str = "local") -> TTSTool:
    return TTSTool(
        api_key="dummy-key",
        voice_id="v1",
        output=output,
        engine=engine,
        sbv2_url="http://127.0.0.1:5001",
    )


# ── Config ────────────────────────────────────────────────────────────────

def test_the_engine_defaults_to_sbv2():
    with patch.dict(os.environ, {}, clear=True):
        assert TTSConfig().engine == "sbv2"


def test_the_engine_can_be_switched_back_to_elevenlabs():
    """SBV2 が動かないときに戻せるようにしておく。"""
    with patch.dict(os.environ, {"TTS_ENGINE": "elevenlabs"}, clear=True):
        assert TTSConfig().engine == "elevenlabs"


# ── 合成の経路 ─────────────────────────────────────────────────────────────

def test_sbv2_synthesises_locally_and_never_calls_elevenlabs():
    """ローカル化の要点。外部 API を叩かない（無料枠を使わない）。"""
    tool = _tool(engine="sbv2")
    with patch.object(tool, "_synth_sbv2", new=AsyncMock(return_value=_WAV)) as synth, \
         patch.object(tool, "_play_paths", new=AsyncMock(return_value=["local"])), \
         patch("aiohttp.ClientSession") as session:
        result = asyncio.run(tool.say("こんばんは"))
    synth.assert_awaited_once()
    assert synth.await_args.args[0] == "こんばんは"
    session.assert_not_called()                     # ElevenLabs は叩かない
    assert "こんばんは" in result


def test_elevenlabs_is_still_reachable_when_selected():
    """従来の経路も残す（engine=elevenlabs）。"""
    tool = _tool(engine="elevenlabs")
    with patch.object(tool, "_synth_sbv2", new=AsyncMock(return_value=_WAV)) as synth, \
         patch.object(tool, "_say_elevenlabs", new=AsyncMock(return_value="Said: x")) as eleven:
        asyncio.run(tool.say("こんばんは"))
    synth.assert_not_awaited()
    eleven.assert_awaited_once()


def test_silent_output_calls_neither_engine():
    """実機テスト用の silent は、どちらの合成も走らせない。"""
    tool = _tool(engine="sbv2", output="silent")
    with patch.object(tool, "_synth_sbv2", new=AsyncMock(return_value=_WAV)) as synth, \
         patch.object(tool, "_say_elevenlabs", new=AsyncMock()) as eleven:
        result = asyncio.run(tool.say("こんばんは"))
    synth.assert_not_awaited()
    eleven.assert_not_awaited()
    assert "silent" in result


def test_a_dead_server_degrades_instead_of_raising():
    """サーバーが落ちていても例外を投げない。話せなかったことだけを返す。"""
    tool = _tool(engine="sbv2")
    with patch.object(tool, "_synth_sbv2", new=AsyncMock(side_effect=OSError("接続できない"))), \
         patch.object(tool, "_play_paths", new=AsyncMock(return_value=["local"])) as play:
        result = asyncio.run(tool.say("こんばんは"))
    play.assert_not_called()
    assert "話せなかった" in result


# ── サーバーの起動 ──────────────────────────────────────────────────────────

def test_the_server_is_started_at_boot_only_when_it_will_be_used():
    """使わない構成では起こさない（GPU と十数秒を使わせない）。"""
    from familiar_agent.tools.tts import ensure_sbv2_server

    with patch("familiar_agent.tools.tts._spawn_sbv2") as spawn, \
         patch("familiar_agent.tools.tts._sbv2_is_alive", return_value=False):
        ensure_sbv2_server(TTSConfig(), engine="sbv2", output="local")
        assert spawn.called
        spawn.reset_mock()

        ensure_sbv2_server(TTSConfig(), engine="elevenlabs", output="local")
        assert not spawn.called          # 別のエンジンを使う構成
        ensure_sbv2_server(TTSConfig(), engine="sbv2", output="silent")
        assert not spawn.called          # 音を出さない構成


def test_an_already_running_server_is_not_started_twice():
    from familiar_agent.tools.tts import ensure_sbv2_server

    with patch("familiar_agent.tools.tts._spawn_sbv2") as spawn, \
         patch("familiar_agent.tools.tts._sbv2_is_alive", return_value=True):
        ensure_sbv2_server(TTSConfig(), engine="sbv2", output="local")
    assert not spawn.called
