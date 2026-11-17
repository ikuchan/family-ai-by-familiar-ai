"""環-aw 第 3 段 項目 2：確認の持ち主は状態図（`設計方針_パジュの状態` v0.15 §2.1、§3.1、決定グ、ズ）。

確認を置く口（`agent.ask_confirm`）は非同期で、状態図の待っている答えへ置く。タイマーとアラームの道具は、その口を
待ってから「まだ掛けていない」を返す。返ったときには、状態図が確認を待っている。読む口（`confirm_alive`、
`confirm_frame`）と解く口（`resolve_confirm`）も状態図を読む。agent は確認の印を持たない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import confirm_state
from familiar_agent.loop.paju_state import PajuState

from tests.test_alarm import _tool as _alarm_tool
from tests.test_timer_tool import _tool as _timer_tool


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "true")


def _answer(sm: PajuState) -> str:
    ids = {s.id for s in sm.configuration}
    return next(
        (n for n in ("no_answer", "confirm_answer", "offer_answer", "heard_answer") if n in ids),
        "",
    )


def _agent(sm: PajuState):
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a.paju_state = sm
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("掛けた：id=1", True))
    a._alarm_tool = MagicMock()
    a._alarm_tool.call = AsyncMock(return_value=("掛けた：id=2", True))
    return a


def test_the_timer_tool_returns_after_the_chart_awaits_the_confirm():
    async def go():
        sm = PajuState()
        await sm.start()
        a = _agent(sm)
        t, store, _ = _timer_tool()
        t._ask = a.ask_confirm
        text, ok = await t.call("set_timer", {"after_minutes": 3, "label": "パスタ"})
        assert ok and "まだ掛けていない" in text and store.active() == []
        assert _answer(sm) == "confirm_answer"
        assert sm.confirm().tool_input == {"after_minutes": 3, "label": "パスタ"}

    asyncio.run(go())


def test_the_alarm_tool_returns_after_the_chart_awaits_the_confirm():
    async def go():
        sm = PajuState()
        await sm.start()
        a = _agent(sm)
        t, store, _, _ = _alarm_tool()
        t._ask = a.ask_confirm
        text, ok = await t.call("set_alarm", {"at": "6:30", "label": "起こす"})
        assert ok and "まだ掛けていない" in text and store.active() == []
        assert _answer(sm) == "confirm_answer"
        assert sm.confirm().action == "set_alarm"

    asyncio.run(go())


def test_the_agent_reads_and_resolves_the_confirm_from_the_chart():
    async def go():
        sm = PajuState()
        await sm.start()
        a = _agent(sm)
        assert not a.confirm_alive() and a.confirm_frame() == ""
        pc = confirm_state.PendingConfirm(
            action="set_timer",
            tool_input={"after_minutes": 3, "label": "パスタ"},
            asked_at=__import__("time").time(),
            text="3 分でいい？",
            what="タイマーを掛ける「パスタ」（3 分）",
        )
        await a.ask_confirm(pc)
        assert a.confirm_alive() and a.confirm_frame() == confirm_state.frame(pc)
        text, ok = await a.resolve_confirm(True, now="t")
        a._timer_tool.call.assert_awaited_once_with(
            "set_timer", {"after_minutes": 3, "label": "パスタ"}, now="t", confirmed=True
        )
        assert ok and _answer(sm) == "no_answer" and not a.confirm_alive()
        assert not hasattr(a, "_pending_confirm") and not hasattr(a, "_confirm_prompted_at")

    asyncio.run(go())
