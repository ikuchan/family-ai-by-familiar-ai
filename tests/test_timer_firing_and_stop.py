"""タイマーが鳴る・止める（知-n）——T の tick・配信ゲートの通り抜け・`/timer stop` の命令・W の枠。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import timer_watch
from familiar_agent.loop.event_loop import InformationProcessing, _query_label
from familiar_agent.loop.request import Request

NOW = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)


def _row(tid, label, *, due, passes_quiet=False, asked_by="パパ"):
    return {
        "id": tid,
        "label": label,
        "due": due,
        "started_at": due - timedelta(minutes=3),
        "fired_at": None,
        "cancelled_at": None,
        "asked_by": asked_by,
        "obs_id": None,
        "passes_quiet": passes_quiet,
    }


def test_due_timers_are_fired_once_and_pushed_as_a_device_request():
    store = MagicMock()
    store.due_now = MagicMock(
        return_value=[
            _row(1, "パスタ", due=NOW - timedelta(seconds=1)),
            _row(2, "起こす", due=NOW - timedelta(minutes=40), passes_quiet=True),
        ]
    )
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    fired = timer_watch.fire_due(store, dif, now=NOW)
    assert fired == 2
    assert [c.args[0] for c in store.mark_fired.call_args_list] == [1, 2]
    calls = dif.device.call_args_list
    assert (
        calls[0].args[0] == "タイマー"
        and "パスタ" in calls[0].args[1]
        and calls[0].kwargs["passes_gate"] is False
    )
    # 40 分遅れ（落ちていた間に過ぎた）は「遅れて」を添え、確かめて掛けたものはゲートを通り抜ける。
    assert "遅れて" in calls[1].args[1] and calls[1].kwargs["passes_gate"] is True


def test_nothing_due_pushes_nothing():
    store = MagicMock()
    store.due_now = MagicMock(return_value=[])
    dif = MagicMock()
    assert timer_watch.fire_due(store, dif, now=NOW) == 0
    dif.device.assert_not_called()


def test_a_request_that_passes_the_gate_is_not_blocked():
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._social_presence_permission = MagicMock(return_value=0.0)  # 誰も居ない
    a._in_quiet_hours = MagicMock(return_value=True)
    ip = InformationProcessing(a)
    ip._req = Request(trigger_kind="機器", passes_gate=True)
    assert ip._delivery_block_reason() == ""
    ip._req = Request(trigger_kind="機器", passes_gate=False)
    assert ip._delivery_block_reason() != ""


def test_the_device_trigger_carries_the_flag_into_the_request():
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._iterate = AsyncMock()
    asyncio.run(ip._begin_device("タイマー", "鳴った", passes_gate=True))
    assert ip._req.trigger_kind == "機器" and ip._req.passes_gate is True


def test_timer_actions_have_distinct_labels_per_input():
    assert (
        _query_label("set_timer", {"label": "パスタ", "after_minutes": 3})
        == "タイマーを掛ける「パスタ」"
    )
    assert _query_label("cancel_timer", {"id": 3}) == "タイマーを止める「3」"
    assert _query_label("start_stopwatch", {"label": "走る"}) == "測り始める「走る」"


def test_the_timer_frame_is_appended_to_the_system_prompt():
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._timer_tool = MagicMock()
    a._timer_tool.frame = MagicMock(
        return_value="[タイマー]\n- id=1 パスタ 残り 2:00（10:02 に鳴る）"
    )
    ip = InformationProcessing(a)
    stable, variable = ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="[反復] 1/5")
    assert "[タイマー]" in variable and "パスタ" in variable
    a._timer_tool.frame = MagicMock(return_value="")
    _stable, variable = ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="[反復] 1/5")
    assert "[タイマー]" not in variable


def test_the_slash_command_stops_without_the_llm():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("止めた：id=3 「パスタ」", True))
    a._handle_timer_command = lambda ui: Agent._handle_timer_command(a, ui)
    reply = asyncio.run(a._handle_timer_command("/timer stop 3"))
    assert reply == "止めた：id=3 「パスタ」"
    a._timer_tool.call.assert_awaited_once_with("cancel_timer", {"id": "3"})
    reply = asyncio.run(a._handle_timer_command("/timer stop"))
    a._timer_tool.call.assert_awaited_with("cancel_timer", {"id": "all"})
    assert asyncio.run(a._handle_timer_command("こんにちは")) is None


def test_the_request_remembers_when_it_began_and_the_lookup_passes_it_to_the_timer():
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("掛けた", True))
    ip = InformationProcessing(a)
    asyncio.run(ip._begin_request(kind="発話", text="1分測って", utterance="1分測って"))
    began = ip._req.began_at
    assert isinstance(began, datetime) and began.tzinfo is not None
    assert (datetime.now(timezone.utc) - began).total_seconds() < 5
    asyncio.run(
        ip._run_lookup_body(
            "set_timer", {"after_minutes": 1, "label": "x"}, "タイマーを掛ける「x」", None, 1
        )
    )
    assert a._timer_tool.call.call_args.kwargs["now"] == began


# ── 音で鳴る（知-n-ろ・2026-09-17）─────────────────────────────────────────────
#
# 「タイマーです」の一言でなく、タイマーらしい音（`sounds/timer_alarm.wav`・自作）を
# `TIMER_RING_SEC` のあいだ繰り返す。止めるのは `cancel_timer`・`/timer stop`・時間切れ。
# 鳴っている間もマイクは閉じない（閉じると「止めて」が届かない）。


def test_a_due_timer_rings_the_sound_when_ring_sec_is_positive():
    store = MagicMock()
    store.due_now = MagicMock(return_value=[_row(1, "パスタ", due=NOW - timedelta(seconds=1))])
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    timer_watch.fire_due(store, dif, now=NOW, ring_sec=30.0, quiet=False, gain=1.5)
    dif.ring.assert_called_once_with(seconds=30.0, gain=1.5)


def test_ring_sec_zero_keeps_the_old_voice_only_behaviour():
    store = MagicMock()
    store.due_now = MagicMock(return_value=[_row(1, "パスタ", due=NOW - timedelta(seconds=1))])
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    timer_watch.fire_due(store, dif, now=NOW, ring_sec=0.0, quiet=False)
    dif.ring.assert_not_called()
    dif.device.assert_called_once()


def test_in_quiet_hours_only_a_confirmed_timer_makes_a_sound():
    store = MagicMock()
    store.due_now = MagicMock(
        return_value=[
            _row(1, "パスタ", due=NOW - timedelta(seconds=1)),
            _row(2, "起こす", due=NOW - timedelta(seconds=1), passes_quiet=True),
        ]
    )
    store.mark_fired = MagicMock(return_value=True)
    dif = MagicMock()
    timer_watch.fire_due(store, dif, now=NOW, ring_sec=30.0, quiet=True)
    assert dif.ring.call_count == 1


def test_the_sound_repeats_until_the_time_is_up_and_does_not_use_the_voice(monkeypatch):
    from familiar_agent.io.dif import DIF

    plays = []

    async def fake_play(path, gain):
        plays.append((path.name, gain))
        await asyncio.sleep(0.01)
        return True

    monkeypatch.setattr("familiar_agent.io.dif._play_wav", fake_play)
    tts = MagicMock()
    tts.call = AsyncMock()
    dif = DIF(tts=tts)

    async def run():
        dif.ring(seconds=0.05, gain=1.5)
        await asyncio.sleep(0.12)
        return dif.ringing

    assert asyncio.run(run()) is False
    assert len(plays) >= 2 and plays[0] == ("timer_alarm.wav", 1.5)
    tts.call.assert_not_awaited()  # 声の口を通らない＝マイクの門（tts_active）が立たない


def test_stop_ring_cuts_the_sound_short(monkeypatch):
    from familiar_agent.io.dif import DIF

    async def fake_play(path, gain):
        await asyncio.sleep(0.01)
        return True

    monkeypatch.setattr("familiar_agent.io.dif._play_wav", fake_play)
    dif = DIF(tts=MagicMock())

    async def run():
        dif.ring(seconds=10.0)
        await asyncio.sleep(0.03)
        assert dif.ringing is True
        dif.stop_ring()
        await asyncio.sleep(0.02)
        return dif.ringing

    assert asyncio.run(run()) is False


def test_cancel_timer_stops_the_sound_even_when_nothing_is_active():
    """鳴った時点で `active` からは外れているので、「止めて」は音を止めるためだけに来る。"""
    from familiar_agent.tools.timer import TimerTool

    store = MagicMock()
    store.active = MagicMock(return_value=[])
    on_cancel = MagicMock()
    tool = TimerTool(
        store=lambda: store,
        oif=MagicMock(),
        speaker=lambda: "",
        quiet=lambda: None,
        silence_active=lambda: False,
        on_cancel=on_cancel,
    )
    reply, ok = asyncio.run(tool._cancel({"id": "all"}))
    on_cancel.assert_called_once()


def test_the_agent_stop_hook_reaches_the_dif():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._info_processing = MagicMock()
    Agent._stop_timer_ring(a)
    a._info_processing._dif.stop_ring.assert_called_once()
