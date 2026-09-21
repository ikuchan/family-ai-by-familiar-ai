"""声の速さと、じっくり読む声（環-u・2026-09-21）。

**ふだんは flash で `speed` 0.9**（既定の 1.0 は速すぎた・本人が聴いて決めた）。
**主LLM が話し、かつ外へ問い合わせた返りから起きた反復のときだけ v3**（漢字を読めるので
ひらがな化を通さない。合成に 3.4 秒かかるが、相手はもう数秒待っている場面である）。
"""

from __future__ import annotations


import pytest

from familiar_agent.config import TTSConfig
from familiar_agent.core.voice_rules import ASYNC_RETURNS, careful_voice


# ── どの声で読むか（純関数）────────────────────────────────────────────────


def test_the_full_llm_after_an_outside_lookup_reads_carefully():
    assert careful_voice("full", frozenset({"search_deferred"}))
    assert careful_voice("full", frozenset({"fetch_deferred"}))
    assert careful_voice("full", frozenset({"family_schedule"}))
    assert careful_voice("full", frozenset({"ask_vault_yusuke"}))


def test_the_short_word_of_the_arbiter_never_reads_carefully():
    assert not careful_voice("light", frozenset({"search_deferred"}))
    assert not careful_voice("action", frozenset({"search_deferred"}))


def test_a_quick_return_is_not_worth_the_wait():
    assert not careful_voice("full", frozenset({"recall"}))
    assert not careful_voice("full", frozenset({"set_timer"}))
    assert not careful_voice("full", frozenset({"see"}))
    assert not careful_voice("full", frozenset())


def test_the_outside_tools_are_named():
    assert {"search_deferred", "fetch_deferred", "house_rules", "family_schedule"} <= ASYNC_RETURNS


# ── 速さと、モデルの選び方 ─────────────────────────────────────────────────


def test_the_speed_and_the_careful_model_have_defaults(monkeypatch):
    monkeypatch.delenv("TTS_SPEED", raising=False)
    monkeypatch.delenv("ELEVENLABS_CAREFUL_MODEL", raising=False)
    cfg = TTSConfig()
    assert cfg.speed == pytest.approx(0.9)
    assert cfg.careful_model == "eleven_v3"
    monkeypatch.setenv("TTS_SPEED", "1.0")
    assert TTSConfig().speed == pytest.approx(1.0)


def _tool():
    from familiar_agent.tools.tts import TTSTool

    return TTSTool(api_key="k", voice_id="v", engine="elevenlabs", output="silent")


def test_the_payload_carries_the_speed():
    t = _tool()
    payload = t._elevenlabs_payload("こんにちは", careful=False)
    assert payload["voice_settings"]["speed"] == pytest.approx(0.9)
    assert payload["model_id"] == "eleven_flash_v2_5"


def test_reading_carefully_uses_v3_and_keeps_the_kanji():
    t = _tool()
    payload = t._elevenlabs_payload("その間は黙ってる", careful=True)
    assert payload["model_id"] == "eleven_v3"
    assert payload["text"] == "その間は黙ってる", "v3 は漢字を読めるのでひらがな化しない"


def test_the_usual_voice_still_gets_the_reading():
    t = _tool()
    payload = t._elevenlabs_payload("その間は黙ってる", careful=False)
    assert payload["text"] != "その間は黙ってる", "flash は漢字を読めないのでひらがなに直す"


@pytest.mark.asyncio
async def test_the_port_passes_the_choice_through():
    from familiar_agent.io.dif import DIF
    from unittest.mock import AsyncMock, MagicMock

    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: ok", None))
    dif = DIF(tts=tts)
    await dif.speak("長い答え", careful=True)
    assert tts.call.await_args.args[1]["careful"] is True


# ── 反復がその判定を使う ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_answer_after_an_outside_lookup_is_read_carefully():
    """調べものの返りで主LLM が答えるときだけ、じっくり読む声（環-u）。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    ip = InformationProcessing(_agent(stream_returns=[_turn([])]))
    ip._returned_now = frozenset({"search_deferred"})
    assert ip._careful_voice("full") is True
    assert ip._careful_voice("light") is False
    ip._returned_now = frozenset({"recall"})
    assert ip._careful_voice("full") is False
