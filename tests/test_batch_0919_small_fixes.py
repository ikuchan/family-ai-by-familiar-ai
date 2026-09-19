"""09-19 の実機で見つけた小さな 5 件（出-ag・出-af・出-ae(1)・知-y・知-u-ろ）。

- 出-ag：道具の帰りの反復で、返りが断りなのに「測り始めますね」と言った（13:39）。帰りの先導文に基準文。
- 出-af：鳴っている最中の「止めて」で音は止めたのに「止めるものが無い」と返した（13:34・昨日 21:55）。
- 出-ae(1)：誰か分からないのに直近の名前で「パパ」と呼んだ（13:14）。在席の注記に「名前で呼ばない」。
- 知-y：Whisper の幻聴の定型文「ご視聴ありがとうございました」が会話入力になった（13:32）。
- 知-u-ろ：ストップウォッチの名前が「ストップウォッチの開始」「時間を測る」（13:38）。行為の語は既定名へ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.core.label_rules import clean_label
from familiar_agent.core.stt_rules import is_hallucination
from familiar_agent.loop import arbiter, generator


def test_the_tool_return_lead_forbids_claiming_what_the_return_denies():
    lead = arbiter._LEAD_TOOL_RETURN
    assert "道具の返りにそう書いてあるときだけ" in lead and "断り" in lead


def test_cancel_reports_that_it_stopped_the_sound():
    from tests.test_timer_tool import _tool

    t, store, _ = _tool()
    t._on_cancel = MagicMock(return_value=True)  # 鳴っていた音を止めた
    text, ok = asyncio.run(t.call("cancel_timer", {"id": "all"}))
    assert ok and "音を止めた" in text
    t._on_cancel = MagicMock(return_value=False)  # 何も鳴っていなかった
    text, ok = asyncio.run(t.call("cancel_timer", {"id": "all"}))
    assert ok and text == "動いているタイマーは無い"


def test_stop_timer_ring_says_whether_it_stopped_anything():
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._info_processing = MagicMock()
    a._info_processing._dif.ringing = True
    assert a._stop_timer_ring() is True
    a._info_processing._dif.stop_ring.assert_called_once()
    a._info_processing._dif.ringing = False
    assert a._stop_timer_ring() is False


def test_an_unconfirmed_presence_says_not_to_use_a_name():
    a = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=[])
    a._pmm.speaker_status = MagicMock(return_value=None)
    a.speaker_known = MagicMock(return_value=False)
    a._presence_sensor.room_occupied = MagicMock(return_value=True)
    ctx = generator._present_ctx(a)
    assert "unconfirmed" in ctx and "名前で呼ばない" in ctx


def test_whisper_boilerplate_is_dropped_but_real_thanks_are_not():
    assert is_hallucination("ご視聴ありがとうございました")
    assert is_hallucination(" ご視聴ありがとうございました。 ")
    assert is_hallucination("ご清聴ありがとうございました")
    assert is_hallucination("ありがとうございました。")
    assert not is_hallucination("教えてくれてありがとうございました")
    assert not is_hallucination("こんにちは")


def test_the_transcriber_drops_the_boilerplate(monkeypatch):
    from familiar_agent.tools import local_stt

    assert local_stt.drop_if_hallucination("ご視聴ありがとうございました。") == ""
    assert local_stt.drop_if_hallucination("パジュ、こんにちは") == "パジュ、こんにちは"


def test_labels_that_name_the_act_fall_back():
    assert clean_label("ストップウォッチの開始", "測る") == "測る"
    assert clean_label("時間を測る", "測る") == "測る"
    assert clean_label("タイマー開始", "タイマー") == "タイマー"
    assert clean_label("お風呂", "測る") == "お風呂"
    assert clean_label("パパの頼み", "タイマー") == "パパの頼み"
