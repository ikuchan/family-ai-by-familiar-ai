"""家族の予定を先読みし、変わったら O に記録する（情-q・2026-10-10・本人の決定）。

BOND で話しかけるとき、W が家族の予定に寄るように、予定を O に置く。T が起動直後に 1 回、そのあと 1 時間に 1 回
`get_family_schedule(days=7)` を読み、時刻の行（`【いま】`・`【出典】`）を外した本文を前回（`agent_state.family_schedule`）と
比べる。変わっていれば機器の記録「予定」として O に書く。**初回も書く**（O に無いと手がかりで引けない）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import schedule_watch

S1 = (
    "【いま】2026-10-10 土曜日 15:00（JST）\n【出典】ファミリーカレンダー（15:00 に取得・今日から 7 日ぶん）\n\n"
    "- 10-11（日） 10:00〜12:00 たいき サッカーの試合"
)
S1_LATER = S1.replace("15:00", "16:00")
S2 = S1 + "\n- 10-13（火） 18:00 こうき 塾"


def _ip(reply: str, ok: bool = True, connected: bool = True):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent as _real_agent

    ip = InformationProcessing(_real_agent(stream_returns=[]))
    ip._dif = MagicMock()
    ip._dif.call_tool = AsyncMock(return_value=(reply, ok))
    ip._dif.tool_defs = MagicMock(
        return_value=[{"name": "get_family_schedule"}] if connected else []
    )
    ip.record_device = AsyncMock()
    return ip


def test_the_first_read_is_recorded_and_stored():
    ip = _ip(S1)
    schedule_watch._save_state(None)
    assert asyncio.run(schedule_watch.check_schedule(ip)) is True
    assert ip._dif.call_tool.await_args.args == ("get_family_schedule", {"days": 7})
    kind, content = ip.record_device.await_args.args[:2]
    assert kind == "予定" and "家族のこれから 7 日の予定" in content and "サッカーの試合" in content
    assert "【いま】" not in content
    assert schedule_watch.stored() == "- 10-11（日） 10:00〜12:00 たいき サッカーの試合"


def test_only_the_clock_changing_is_not_a_change():
    ip = _ip(S1_LATER)
    schedule_watch._save_state("- 10-11（日） 10:00〜12:00 たいき サッカーの試合")
    assert asyncio.run(schedule_watch.check_schedule(ip)) is False
    ip.record_device.assert_not_awaited()


def test_a_changed_schedule_is_recorded():
    ip = _ip(S2)
    schedule_watch._save_state("- 10-11（日） 10:00〜12:00 たいき サッカーの試合")
    assert asyncio.run(schedule_watch.check_schedule(ip)) is True
    assert "こうき 塾" in ip.record_device.await_args.args[1]
    assert "こうき 塾" in (schedule_watch.stored() or "")


def test_a_missing_or_failing_tool_does_nothing():
    ip = _ip(S1, connected=False)
    assert asyncio.run(schedule_watch.check_schedule(ip)) is False
    ip._dif.call_tool.assert_not_called()
    ip = _ip("エラー", ok=False)
    assert asyncio.run(schedule_watch.check_schedule(ip)) is False
    ip.record_device.assert_not_awaited()


def test_the_tonic_reads_at_start_and_once_an_hour():
    import inspect

    from familiar_agent.loop.tonic import Tonic

    assert schedule_watch.INTERVAL_SEC == 3600.0
    assert "check_schedule(self._ip)" in inspect.getsource(Tonic._maybe_check_schedule)
    t = Tonic.__new__(Tonic)
    t._schedule_checked_at = None
    assert t._schedule_due(now=100.0) is True
    t._schedule_checked_at = 100.0
    assert t._schedule_due(now=100.0 + 3599) is False
    assert t._schedule_due(now=100.0 + 3600) is True
