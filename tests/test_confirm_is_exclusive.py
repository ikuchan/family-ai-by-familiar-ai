"""確認待ちのあいだは、その確認だけが動く（知-o・2026-10-13・本人の決定）。

10/13 20:00:25 に、確認待ちのあいだ情動（safety）が「出入口を見に行く」を始め、その完了が会話の求めを閉じた後に届いて
調停を回し、タイマー中に「黙っていてもよろしいですか？」と話した（20:00:41）。確認待ちのあいだは：
- 情動の求めを立てない。
- 対応する意図が無い完了では反復を起こさない。
- 名前の無い言葉も前の求めに添えず、新しい会話の求め（3 択）にする。
- 機器（鳴ったタイマー・アラームの知らせ）は今のまま通す。
- 調停へ確認の問い・決まった促し・促す間隔が空いたかを渡し、促したら時刻を残す。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from familiar_agent.core.confirm_state import PendingConfirm
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger
from familiar_agent.loop.arbiter import Decision

from tests.test_event_loop import _agent

ASKED = "3 分のタイマーね。その間は黙って聞かないよ、いい？"


def _ip(*, confirming: bool = True):
    a = _agent(stream_returns=[])
    pc = (
        PendingConfirm(action="set_timer", asked_at=time.time(), text=ASKED) if confirming else None
    )
    a._pending_confirm = pc
    a._confirm_prompted_at = pc.asked_at if pc else None
    a.confirm_alive = lambda: a._pending_confirm is not None
    ip = InformationProcessing(a)
    return ip, a


def _drive_once(ip, trigger: "Trigger | None", **patches) -> dict:
    """駆動体を 1 回だけ回し、どの口へ渡したかを記録する。"""
    seen: dict = {}

    async def go():
        async def take():
            if "taken" in seen:
                raise asyncio.CancelledError
            seen["taken"] = True
            return trigger

        ip._take_trigger = take  # type: ignore[method-assign]
        ip._swallow_if_unheard = AsyncMock(return_value=False)  # type: ignore[method-assign]
        for name in (
            "_begin_affect",
            "_begin_device",
            "_begin_utterance",
            "_attach_to_request",
            "_iterate",
        ):
            setattr(
                ip, name, AsyncMock(side_effect=lambda *a, n=name, **k: seen.setdefault("to", n))
            )
        for k, v in patches.items():
            setattr(ip, k, v)
        try:
            await ip._drive()
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    return seen


def test_an_affect_does_not_start_while_confirming():
    ip, _ = _ip()
    assert "to" not in _drive_once(ip, Trigger(kind="情動", query="safety", result="見回りたい"))


def test_an_affect_starts_without_a_confirmation():
    """**反証の側**：確認待ちが無ければ、情動で求めが立つ。"""
    ip, _ = _ip(confirming=False)
    assert (
        _drive_once(ip, Trigger(kind="情動", query="safety", result="見回りたい"))["to"]
        == "_begin_affect"
    )


def test_a_device_still_starts_while_confirming():
    ip, _ = _ip()
    assert (
        _drive_once(ip, Trigger(kind="機器", query="タイマー", result="鳴った"))["to"]
        == "_begin_device"
    )


def test_an_orphan_completion_does_not_iterate_while_confirming():
    ip, _ = _ip()
    ip._drained_completions.append(Trigger(kind="完了", query="出入口を見に行く", result="見た"))
    assert "to" not in _drive_once(ip, None)
    assert ip._drained_completions == []


def test_an_owned_completion_still_iterates_while_confirming():
    """**反証の側**：確認待ちを置いた求めの完了（開いた意図がある）は取り込む（求めを黙って閉じるため）。"""
    ip, _ = _ip()
    ip._req.lookups.append(
        Lookup(index=1, action="set_timer", query="タイマーを掛ける", generation=0)
    )
    ip._drained_completions.append(
        Trigger(kind="完了", query="タイマーを掛ける", result="まだ掛けていない。")
    )
    assert _drive_once(ip, None)["to"] == "_iterate"


def test_an_unnamed_word_is_not_attached_while_confirming():
    ip, _ = _ip()
    ip._req.lookups.append(
        Lookup(index=1, action="set_timer", query="タイマーを掛ける", generation=0)
    )
    seen = _drive_once(ip, Trigger(kind="会話入力", query="はい", named=False))
    assert seen["to"] == "_begin_utterance"


def test_the_arbiter_gets_the_confirmation():
    ip, a = _ip()
    a._confirm_prompted_at = time.time() - 30
    got = ip._arbiter_input(utterance="はい", workspace_ctx="", present_ctx="")
    assert got.confirm_question == ASKED
    assert got.confirm_prompt.startswith("いまタイマーを掛けるか")
    assert got.confirm_prompt_due


def test_the_prompt_waits_after_the_question():
    """**反証の側**：問いを出してすぐ（20 秒未満）は促さない。確認待ちが無ければ問いを渡さない。"""
    ip, _ = _ip()
    assert not ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").confirm_prompt_due
    ip2, _ = _ip(confirming=False)
    assert (
        ip2._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").confirm_question == ""
    )


def test_a_prompt_is_stamped():
    ip, a = _ip()
    a._confirm_prompted_at = 0.0
    prompt = ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").confirm_prompt
    ip._note_confirm_prompt(Decision(branch="light", text=prompt))
    assert a._confirm_prompted_at > 0.0
    a._confirm_prompted_at = 0.0
    ip._note_confirm_prompt(Decision(branch="light", text="ほかの返事"))
    assert a._confirm_prompted_at == 0.0  # 促しでなければ時刻を動かさない


def test_asking_a_confirmation_starts_the_gap():
    from familiar_agent.agent import EmbodiedAgent

    a = _agent(stream_returns=[])

    pc = PendingConfirm(action="set_timer", asked_at=123.0, text=ASKED)
    EmbodiedAgent.ask_confirm(a, pc)
    assert a._confirm_prompted_at == 123.0
