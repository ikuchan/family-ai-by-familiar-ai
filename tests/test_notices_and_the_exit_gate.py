"""知らせでは話しかけず、発話の門は情動だけを在席で止める（出-as 段 6・2026-09-26・`設計方針_話していいかの決まり` §2.1・§2.7）。

- 人が来た・居なくなった・メモ：**求めを立てずに記憶へ記録だけする**（`record_device`）。話しかけない。
- 黙るのが明けた：何もしない（黙っていたあいだに聞いたことは次の会話の求めに乗る）。
- 発話の門：会話の求めは在席で止めない（窓で決まっている）。情動は在席で止める。静穏時間の門は外す。
- 止められた発話は、すべて独り言にする（保留して後で配るのはやめる）。
- タイマーの操作の言葉を受けたときも窓を開ける（返事「止めたよ」を声にするため）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.silence_state import SilenceRequest

from tests.test_event_loop import _agent


def _ip(kind="発話", *, present=1.0, quiet=False):
    a = _agent(stream_returns=[])
    a._social_presence_permission = MagicMock(return_value=present)
    a._in_quiet_hours = MagicMock(return_value=quiet)
    ip = InformationProcessing(a)
    ip._req.trigger_kind = kind
    ip._req.fired_axis = "bond"  # 情動なら話しかける軸（出-as 段 8：話すのは bond・esteem だけ）
    return a, ip


# ── 知らせは記録だけ ───────────────────────────────────────────────────────


def test_a_notice_is_recorded_without_starting_a_request():
    a, ip = _ip()
    asyncio.run(ip.record_device("入室", "パパ が来た"))
    assert ip._triggers.empty()  # 求めを立てない
    written = [c.args[0] for c in a._memory.save_async_with_id.call_args_list]
    assert any("[入室] パパ が来た" in str(w) for w in written)


def test_the_memo_watcher_records_instead_of_notifying(monkeypatch):
    from familiar_agent.loop import notes_watch

    ip = MagicMock()
    ip._dif.tool_defs = MagicMock(return_value=[{"name": notes_watch.TOOL}])
    ip._dif.call_tool = AsyncMock(return_value=("本文：パパへ：牛乳を買う", True))
    monkeypatch.setattr(notes_watch, "_load_state", lambda: "本文：")
    monkeypatch.setattr(notes_watch, "_save_state", lambda body: None)
    monkeypatch.setattr(notes_watch, "body_of", lambda text: text)
    ip.record_device = AsyncMock()
    assert asyncio.run(notes_watch.check_notes(ip)) is True
    ip.record_device.assert_awaited_once()
    assert ip.record_device.await_args.args[0] == "メモ"
    ip._dif.device.assert_not_called()


def test_a_lifted_silence_does_not_start_a_request(monkeypatch):
    from familiar_agent.core.silence_hold import Heard

    a, ip = _ip()
    ip.push_device = MagicMock()
    ip._load_silence = lambda: SilenceRequest(person="パパ", until=time.time() - 1)
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: None)
    ip._muted = [Heard(kind="会話入力", text="明日の予定は？", at=time.time())]
    ip.check_silence_lifted()
    ip.push_device.assert_not_called()


# ── 発話の門 ───────────────────────────────────────────────────────────────


def test_a_conversation_is_not_stopped_by_absence():
    _, ip = _ip("発話", present=0.0)
    assert ip._delivery_block_reason() == ""


def test_affect_is_stopped_by_absence():
    _, ip = _ip("情動", present=0.0)
    assert ip._delivery_block_reason() == "聞く相手が居ない"


def test_quiet_hours_no_longer_stop_speech():
    _, ip = _ip("情動", present=1.0, quiet=True)
    assert ip._delivery_block_reason() == ""


def test_a_blocked_utterance_becomes_a_monologue_not_held():
    a, ip = _ip("情動", present=0.0)
    spoken, outcome = asyncio.run(ip._speak("ねえねえ"))
    assert outcome == "独白"


def test_a_blocked_device_utterance_is_a_monologue_too():
    a, ip = _ip("機器", present=0.0)
    ip._req.request_text = "[音楽] 30 分たった"
    ip._delivery_block_reason = lambda: "聞く相手が居ない"
    spoken, outcome = asyncio.run(ip._speak("音楽を止めたよ"))
    assert outcome == "独白"


# ── タイマーの操作の言葉は窓を開ける ─────────────────────────────────────


def _timer_ip(*, silence=None):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a._timer_tool.frame = MagicMock(return_value="[タイマー]\n- id=1 パスタ 鳴っている")
    a._dif.ringing = True
    a._social_presence_permission = MagicMock(return_value=1.0)
    ip = InformationProcessing(a)
    ip._load_silence = lambda: silence
    return ip


def _admit(ip, text):
    async def go():
        return await ip._swallow_if_unheard(Trigger(kind="会話入力", query=text, source="voice"))

    return not asyncio.run(go())


def test_a_timer_control_word_opens_the_window():
    ip = _timer_ip()
    assert _admit(ip, "止めて") is True
    assert ip._wake_window().is_open(time.monotonic())


def test_a_timer_control_word_under_a_timer_silence_opens_the_window():
    ip = _timer_ip(silence=SilenceRequest(person="", until=time.time() + 300, reason="timer:1"))
    assert _admit(ip, "止めて") is True
    assert ip._wake_window().is_open(time.monotonic())
