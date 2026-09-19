"""普段の声の倍率と、声に出すときの読み（環-q-ろ・出-ac・2026-09-19）。

- ミキサーは 1 つ（YVC-300 `PCM` 80%）で音と声が同じ出口。音（倍率 1.5）を基準にすると声は大きすぎた。
  普段の声は `TTS_GAIN`（既定 0.25・聴き比べで決めた）を掛け、タイマー／アラームの知らせだけ `TIMER_VOICE_GAIN`（1.5）。
- ElevenLabs が「出入口」を「しゅつにゅうくち」と読む。声にする直前に読みの表で置き換える（画面の文字はそのまま）。
- 鳴る音の長さは 30 秒のまま。8 秒で小さく・20 秒で戻す山谷は `test_ring_envelope`。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.config import AgentConfig
from familiar_agent.core.reading import for_speech
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_defaults_are_the_values_we_chose(monkeypatch):
    for k in ("TTS_GAIN", "TIMER_RING_SEC", "ALARM_RING_SEC"):
        monkeypatch.delenv(k, raising=False)
    cfg = AgentConfig()
    assert (
        cfg.tts_gain == 0.25 and cfg.timer_ring_sec == 30.0 and cfg.alarm_ring_sec == 30.0
    )  # 長さは 30（山谷は `test_ring_envelope`）


def test_ordinary_speech_uses_tts_gain_and_notices_keep_the_timer_gain():
    a = _agent(stream_returns=[])
    a.config = MagicMock()
    a.config.tts_gain = 0.25
    a.config.timer_voice_gain = 1.5
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    assert ip._voice_gain() == 0.25
    ip._req.trigger_kind = "機器"
    ip._req.request_text = "[タイマー] タイマー：「パスタ」の時間"
    assert ip._voice_gain() == 1.5
    ip._req.request_text = "[アラーム] アラーム：「起こす」の時刻"
    assert ip._voice_gain() == 1.5


def test_readings_replace_only_in_speech():
    assert for_speech("出入口を見てくるね") == "でいりぐちを見てくるね"
    assert for_speech("窓のほうを見る") == "窓のほうを見る"


def test_the_tts_applies_readings_before_speaking():
    from familiar_agent.tools.tts import TTSTool

    t = TTSTool("k", "v", engine="elevenlabs")
    assert "でいりぐち" in t._clean_for_speech("出入口へ（首を回す）")
