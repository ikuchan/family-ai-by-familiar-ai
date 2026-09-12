"""つなぎを出したなら本応答も出す（環-i）。

つなぎと本応答が別々に配信ゲートを引くので、あいだで在席の証拠が切れると「見てみますね」
だけ出て本文が保留（人が起点）／独白（欲求が起点）になった（実機 2026-09-12 21:21）。
一度つなぎを口に出した求めでは、「聞く相手が居ない」「静穏時間である」で本応答を止めない。
例外は「黙っているよう頼まれている」だけ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, trigger_kind: str, blocked: str, said_filler: bool):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = trigger_kind
    if said_filler:
        ip._req.said_fillers.append("見てみますね")
    ip._delivery_block_reason = lambda: blocked  # type: ignore[method-assign]
    ip._hold_speech = AsyncMock()  # type: ignore[method-assign]
    ip._dif = MagicMock(speak=AsyncMock())
    return ip


def _speak(ip, text="部屋には机と椅子があるよ。"):
    async def scenario():
        got = await ip._speak(text)
        await ip.close()
        return got, ip._dif.speak.await_count, ip._hold_speech.await_count

    return asyncio.run(scenario())


def test_after_a_filler_the_answer_is_spoken_even_if_nobody_is_detected() -> None:
    got, spoke, held = _speak(
        _ip(trigger_kind="発話", blocked="聞く相手が居ない", said_filler=True)
    )
    assert got == ("部屋には机と椅子があるよ。", "発話") and spoke == 1 and held == 0


def test_after_a_filler_quiet_hours_do_not_stop_the_answer() -> None:
    got, spoke, _ = _speak(_ip(trigger_kind="情動", blocked="静穏時間である", said_filler=True))
    assert got[1] == "発話" and spoke == 1


def test_a_monologue_after_a_filler_is_spoken_too() -> None:
    got, spoke, _ = _speak(_ip(trigger_kind="情動", blocked="聞く相手が居ない", said_filler=True))
    assert got[1] == "発話" and spoke == 1


def test_being_asked_to_stay_quiet_still_wins_after_a_filler() -> None:
    got, spoke, held = _speak(
        _ip(trigger_kind="発話", blocked="黙っているよう頼まれている", said_filler=True)
    )
    assert got == ("", "保留") and spoke == 0 and held == 1


def test_without_a_filler_the_gate_works_as_before() -> None:
    got, spoke, held = _speak(
        _ip(trigger_kind="発話", blocked="聞く相手が居ない", said_filler=False)
    )
    assert got == ("", "保留") and spoke == 0 and held == 1
