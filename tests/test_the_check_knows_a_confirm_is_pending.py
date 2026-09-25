"""発話前の検査に、確認待ちかどうかを機械の事実として渡す（出-ag-ろ 穴 4・2026-09-25）。

`2b75f71`（2026-09-21）で規則 `no-claim-while-confirming`（[確認待ち] の枠があるあいだは「掛けた」
「セットした」と言わない）を検査へ渡した。ところが検査が受け取るのは直近のやりとりと
`facts_ctx` だけで、**どちらにも確認待ちかどうかが無かった**。検査の指示は「事実に無いことは、
反しているとも反していないとも言えない」なので、この規則だけはいつも判定できなかった。

確認待ちかどうかはループが知っている（`agent.confirm_frame`）。見たかどうかと同じく、
推測の要らない事実として渡す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.speech_check import facts_ctx

from tests.test_event_loop import _agent

FRAME = (
    "[確認待ち] タイマーを掛ける「タイマー」（3 分）：「3 分のタイマーね。その間は黙って"
    "聞かないよ、いい？」——「いい」「うん」なら confirm、「やめて」「いらない」なら decline"
)


def test_the_facts_say_a_confirm_is_pending():
    ctx = facts_ctx(saw=False, memories=[], confirming=FRAME)
    assert "確認待ちの預かり：あり" in ctx
    assert "3 分のタイマーね" in ctx


def test_the_facts_say_none_is_pending():
    """反証：預かりが無ければ「なし」と書く（黙って省かない。無いことも事実である）。"""
    assert "確認待ちの預かり：なし" in facts_ctx(saw=False, memories=[])


def _facts_handed_over(frame: str) -> str:
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    a._evaluator.check_speech = AsyncMock(return_value=None)
    a.confirm_frame = MagicMock(return_value=frame)
    ip = InformationProcessing(a)

    async def scenario():
        await ip._speech_check_violation("はい、3分ですね。タイマーをセットしました。", "", [])
        await ip.close()
        return a._evaluator.check_speech.call_args.kwargs["facts"]

    return asyncio.run(scenario())


def test_the_loop_hands_the_pending_confirm_to_the_check():
    facts = _facts_handed_over(FRAME)
    assert "確認待ちの預かり：あり" in facts
    assert "3 分のタイマーね" in facts


def test_the_loop_says_none_when_nothing_is_pending():
    assert "確認待ちの預かり：なし" in _facts_handed_over("")
