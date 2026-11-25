"""端-b 段 1-ろ-2：端末の求めの返事は、家で声にせず、話しかけた端末へ返す（課題9 §4 の 3・b1・b2）。

- 返事の字は端末の受け手（`set_device_listener`）へ送り、家のスピーカーでも画面でも出さない。
- 考え中の合図は家で鳴らさず、「いま何の合図か」だけを端末へ送る（b1）。
- 家で音が出る道具（タイマーとアラームの止め方・ストップウォッチ・音楽・印刷）と確認の答えは、端末の求めでは使わない（仮）。
  タイマーとアラームを掛けるのは、端末の時計アプリへ頼む（端-c c6a・`tests/test_device_clock.py`）。
- 求めが閉じたら端末へ知らせる（返事の無いまま閉じても、端末が待ち続けない）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.actions import _HOME_ONLY_ACTIONS
from familiar_agent.loop.event_loop import InformationProcessing

pytestmark = pytest.mark.real_window

TAB = "tablet-1"


def _ip(device: str = TAB):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a.config.tts_gain = 1.0
    a._pmm.presence_status = MagicMock(return_value=[])
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock(return_value="ok")
    ip._dif.say_recorded = AsyncMock(return_value=True)
    ip._dif.beep = AsyncMock(return_value=True)
    ip._req.trigger_kind = "発話"
    ip._req.device = device
    got: list = []
    ip.set_device_listener(lambda dev, kind, payload: got.append((dev, kind, payload)))
    shown: list = []
    ip.set_output(lambda text: shown.append(text), None)
    return ip, got, shown


def test_a_device_reply_is_not_voiced_at_home():
    ip, _, _ = _ip()
    assert asyncio.run(ip._say_aloud("晴れだよ")) is True
    assert asyncio.run(ip._say_aloud("時間切れ", recording="timeout")) is True
    ip._dif.speak.assert_not_called()
    ip._dif.say_recorded.assert_not_called()


def test_a_device_reply_goes_to_the_device_and_not_to_the_home_screen():
    ip, got, shown = _ip()
    ip._emit("晴れだよ")
    assert got == [(TAB, "say", {"text": "晴れだよ", "display": "晴れだよ"})]
    assert shown == []


def test_a_home_reply_is_still_voiced_and_shown_at_home():
    """**反証の側**：家の求めは、いままでどおり家で声にして画面に出し、端末へは送らない。"""
    ip, got, shown = _ip(device="")
    asyncio.run(ip._say_aloud("晴れだよ"))
    ip._emit("晴れだよ")
    ip._dif.speak.assert_awaited_once()
    assert shown == ["晴れだよ"] and got == []


def test_home_sound_tools_are_not_offered_to_a_device_request():
    ip, _, _ = _ip()
    ip._gated = lambda defs: defs or [{"name": "x"}]
    offered = set(ip._extra_actions())
    assert not offered & set(_HOME_ONLY_ACTIONS)
    assert "house_rules" in offered  # 音の出ない家の道具は使える
    ip._req.device = ""
    assert "set_timer" in set(ip._extra_actions())


def test_home_sound_tool_defs_are_not_given_to_the_main_llm_for_a_device():
    ip, _, _ = _ip()
    ip._ACTIONS = {
        "cancel_timer": lambda _ip: [{"name": "cancel_timer"}],
        "say": lambda _ip: [{"name": "say"}],
    }
    ip._gated = lambda defs: defs
    names = {d["name"] for d in ip._tools(actions=("say", "cancel_timer"), cache_tools=False)}
    assert names == {"say"}


def test_a_home_sound_tool_called_for_a_device_is_refused():
    ip, _, _ = _ip()
    ip._agent._timer_tool.call = AsyncMock(return_value=("掛けた", True))
    asyncio.run(ip._run_lookup_body("cancel_timer", {"id": "all"}, "止めて", None))
    ip._agent._timer_tool.call.assert_not_called()
    done = ip._triggers.get_nowait()
    assert done.kind == "完了" and done.failed


def test_the_waiting_cue_goes_to_the_device_once_per_kind():
    ip, got, _ = _ip()
    ip._agent.config.wait_beep_interval_seconds = 0.0
    kinds = iter(["a", "a", "b"])

    def child():
        try:
            return next(kinds)
        except StopIteration:
            ip._request_generation += 1  # 求めが閉じた
            return "b"

    ip._paju().thinking_child = child
    ip._never_speaks = lambda: False
    asyncio.run(asyncio.wait_for(ip._beep_while_waiting(ip._request_generation), timeout=2.0))
    ip._dif.beep.assert_not_called()
    assert [p for _, k, p in got if k == "cue"] == [{"kind": "a"}, {"kind": "b"}]


def test_the_waiting_cue_is_sent_again_after_a_reply_to_the_device():
    """端-c c4 の見直し：端末は返事が届くと合図を止めるので、返事のあとも同じ子で考え続けるなら、もう一度送る。"""
    ip, got, _ = _ip()
    ip._agent.config.wait_beep_interval_seconds = 0.0
    calls = {"n": 0}

    def child():
        calls["n"] += 1
        if calls["n"] == 2:
            ip._to_device("say", {"text": "ちょっと調べるね", "display": "ちょっと調べるね"})
        if calls["n"] > 3:
            ip._request_generation += 1  # 求めが閉じた
        return "a"

    ip._paju().thinking_child = child
    ip._never_speaks = lambda: False
    asyncio.run(asyncio.wait_for(ip._beep_while_waiting(ip._request_generation), timeout=2.0))
    assert [k for _, k, _ in got] == ["cue", "say", "cue"]


def test_the_device_is_told_when_its_request_closes():
    ip, got, _ = _ip()
    ip._tell_device_closed()
    assert (TAB, "closed", {}) in got
    got.clear()
    ip._req.device = ""
    ip._tell_device_closed()
    assert got == []


def test_a_device_word_is_not_shown_as_heard_on_the_home_screen():
    ip, _, _ = _ip()
    heard: list = []
    ip.set_heard_listener(lambda text, ok: heard.append(text))
    ip._notify_heard("明日の天気は？", True, device=TAB)
    ip._notify_heard("おはよう", True)
    assert heard == ["おはよう"]


def test_a_device_asking_for_silence_is_told_home_is_not_silenced():
    """端末の「黙って」は家を黙らせないので、「静かにするね」と約束しない（b4・1-ろ-1 の続き）。"""
    from dataclasses import dataclass

    @dataclass
    class D:
        branch: str = "light"
        text: str = "わかった、静かにするね"
        silence_minutes: int = 15

    ip, _, _ = _ip()
    out = ip._with_silence_reply(D(), None)
    assert "静かにする" not in out.text
    ip._req.device = ""
    assert ip._with_silence_reply(D(), None).text == "わかった、静かにするね"


def test_the_device_gets_the_reply_without_stage_directions():
    ip, got, _ = _ip()
    ip._emit("（にっこり）晴れだよ")
    _, _, payload = got[0]
    assert "にっこり" not in payload["text"] and "にっこり" not in payload["display"]
    assert "晴れだよ" in payload["display"]


def test_the_device_is_not_offered_a_deep_search_after_its_request_closed():
    ip, _, _ = _ip()
    ip._req.spent_actions.add("search_deferred")
    ip._req.request_text = "明日の天気"
    ip._deep_available = lambda: True
    ip._paju().offer_waiting = lambda: False
    assert ip._deep_offer_material() is None
    ip._req.device = ""
    assert ip._deep_offer_material() is not None


def test_the_device_cue_is_sent_while_home_is_sounding():
    ip, got, _ = _ip()
    ip._agent.config.wait_beep_interval_seconds = 0.0
    ip._sounding = lambda: True
    calls = {"n": 0}

    def child():
        calls["n"] += 1
        if calls["n"] > 1:
            ip._request_generation += 1
        return "a"

    ip._paju().thinking_child = child
    ip._never_speaks = lambda: False
    asyncio.run(asyncio.wait_for(ip._beep_while_waiting(ip._request_generation), timeout=2.0))
    assert [p for _, k, p in got if k == "cue"] == [{"kind": "a"}]


def test_closed_is_not_sent_again_by_an_abort_after_the_request_closed():
    ip, got, _ = _ip()
    ip._req.request_id = None
    ip._background_tasks.add(MagicMock())
    ip._abort_lookups = AsyncMock()
    asyncio.run(ip.abort_current())
    assert not [k for _, k, _ in got if k == "closed"]
    ip._req.request_id = "obs-1"
    asyncio.run(ip.abort_current())
    assert [k for _, k, _ in got if k == "closed"] == ["closed"]


def test_a_request_that_ends_without_closing_still_tells_the_device():
    ip, got, _ = _ip()
    ip._to_state = AsyncMock()
    ip._req.request_id = "obs-1"
    asyncio.run(ip._hear_again())
    assert [k for _, k, _ in got] == ["closed"]


def test_an_abort_tells_the_device_it_had_before_waiting():
    """打ち切りを待つあいだに次の求め（家）が始まっても、閉じたと知らせるのは打ち切った求めの端末。"""
    ip, got, _ = _ip()
    ip._req.request_id = "obs-1"

    async def next_request_begins():
        ip._req.device = ""

    ip._abort_lookups = next_request_begins
    asyncio.run(ip.abort_current())
    assert got == [(TAB, "closed", {})]


def test_a_device_saying_you_may_talk_is_told_home_stays_quiet():
    from dataclasses import dataclass

    @dataclass
    class D:
        branch: str = "light"
        text: str = "わかった、話すね"
        silence_minutes: int = 0
        lift_silence: bool = True

    ip, _, _ = _ip()
    ip._paju().silence_request = lambda: object()
    assert "話すね" not in ip._with_silence_reply(D(), None).text
    # 調停が解くと読まなくても、「話していいよ」の言葉そのもので（家の道と同じ・is_release）
    assert (
        "話すね"
        not in ip._with_silence_reply(D(lift_silence=False), None, utterance="話していいよ").text
    )
    # 家が黙っていなければ、「解けない」とは言わない
    ip._paju().silence_request = lambda: None
    assert ip._with_silence_reply(D(), None).text == "わかった、話すね"


def test_no_home_reaction_cue_for_a_device_request():
    ip, _, _ = _ip()
    ip._req.source = "voice"  # 端末の書き起こしを声として渡しても、家では鳴らさない
    assert ip._start_reaction_cue(()) is None
    ip._dif.start_thinking.assert_not_called()


def test_a_closed_request_is_not_closed_again_by_a_later_failure():
    ip, got, _ = _ip()
    ip._to_state = AsyncMock()
    ip._req.request_id = None  # もう閉じた（閉じたあとの例外）
    asyncio.run(ip._hear_again())
    assert got == []
