"""確認待ちのあいだは、掛け直しを機械が落とす（出-ag-ろ・2026-09-21）。

実機 17:34：1 回の「3 分測って」で調停が `set_timer` を **3 回**投げ、そのたびに確認待ちを
作り直し、最後に「タイマーをセットしました」と言った（**掛かっていない**）。W には
`[確認待ち]` の枠が載っていたのに、調停はそれを読まずに投げ直した。
**禁じたい動作は、言葉で頼まず機械が落とす**（本人の決定）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from familiar_agent.core.confirm_state import CONFIRM_TTL_SEC, PendingConfirm, blocks


def _pc(now: float = 1000.0) -> PendingConfirm:
    return PendingConfirm(
        action="set_timer",
        tool_input={"minutes": 3},
        asked_at=now,
        text="3 分のタイマーね。その間は黙って聞かないよ、いい？",
        what="タイマーを掛ける「タイマー」（3 分）",
    )


# ── 何を止めるか（純関数）──────────────────────────────────────────────────


def test_setting_again_while_confirming_is_blocked():
    pc = _pc()
    for action in ("set_timer", "set_alarm", "start_stopwatch"):
        assert blocks(action, pc, now=1001.0), action


def test_answering_and_stopping_always_pass():
    pc = _pc()
    for action in ("confirm", "decline", "cancel_timer", "stop_stopwatch", "cancel_alarm"):
        assert not blocks(action, pc, now=1001.0), action


def test_talking_and_looking_pass():
    pc = _pc()
    for action in ("say", "recall", "see", "look", "search_deferred"):
        assert not blocks(action, pc, now=1001.0), action


def test_nothing_is_blocked_without_a_pending_confirm():
    assert not blocks("set_timer", None, now=1001.0)


def test_an_expired_confirm_blocks_nothing():
    pc = _pc()
    assert not blocks("set_timer", pc, now=1000.0 + CONFIRM_TTL_SEC + 1)


# ── 道具の入口で落ちる ─────────────────────────────────────────────────────


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([])])
    ip = InformationProcessing(a)
    a._pending_confirm = _pc(time.time())  # いま聞いたばかりの預かり
    a.confirm_alive = MagicMock(return_value=True)
    a.confirm_frame = MagicMock(return_value="[確認待ち] タイマーを掛ける「タイマー」（3 分）")
    return ip, a


def test_the_tool_is_not_run_while_confirming():
    from unittest.mock import AsyncMock

    ip, a = _ip()
    a._timer_tool.call = AsyncMock(return_value=("掛けた", True))
    out = _run(ip, "set_timer", {"minutes": 3})
    assert "確かめている" in out, out
    a._timer_tool.call.assert_not_awaited()


def test_answering_still_reaches_the_tool():
    from unittest.mock import AsyncMock

    ip, a = _ip()
    a.resolve_confirm = AsyncMock(return_value=("3 分のタイマーを掛けた", True))
    out = _run(ip, "confirm", {})
    a.resolve_confirm.assert_awaited()
    assert "掛けた" in out


def _run(ip, action: str, tool_input: dict) -> str:
    async def go():
        await ip._run_lookup_body(action, tool_input, "しらべ", None)
        return ip._triggers.get_nowait().result

    return asyncio.run(go())


# ── 確認の問いは機械が出す ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_question_is_spoken_by_the_machine():
    """枠に用意された文をそのまま読む（LLM に作らせない・本人の決定）。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([])])
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    ip._dif.speak = _async_noop()
    await ip._ask_confirm_aloud(_pc())
    ip._dif.speak.assert_awaited_once()
    assert ip._dif.speak.await_args.args[0] == "3 分のタイマーね。その間は黙って聞かないよ、いい？"


def _async_noop():
    from unittest.mock import AsyncMock

    return AsyncMock(return_value=None)


# ── 掛けていないのに掛けたと言わせない ─────────────────────────────────────


def test_the_rule_forbids_claiming_while_confirming():
    from familiar_agent.loop.prompt import CHECKER_RULE_IDS, rules_section

    rules = rules_section()
    assert "no-claim-while-confirming" in rules
    assert "確かめている" in rules
    # 文だけで判じられるので、整合チェックへ渡す
    assert "no-claim-while-confirming" in CHECKER_RULE_IDS


@pytest.mark.asyncio
async def test_a_fresh_confirm_is_asked_aloud_right_away():
    """道具が確認待ちを作ったら、その場で問いを出す（人が答えられるように）。"""
    from unittest.mock import AsyncMock

    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([])])
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock(return_value=None)

    pc = _pc(time.time())

    async def fake_call(action, tool_input, **kw):
        a._pending_confirm = pc  # 道具が預かりを置いた
        return "3 分でいい？", True

    a._timer_tool.call = fake_call
    a._pending_confirm = None
    await ip._run_lookup_body("set_timer", {"minutes": 3}, "タイマー", None)
    ip._dif.speak.assert_awaited_once()
    assert ip._dif.speak.await_args.args[0] == pc.text
