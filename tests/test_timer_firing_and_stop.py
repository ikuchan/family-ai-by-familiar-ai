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
    asyncio.run(ip._begin_device("タイマー", "鳴った", False, passes_gate=True))
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
