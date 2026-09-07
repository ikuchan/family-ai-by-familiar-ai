"""続き先の判定を、軽量LLM の専用の仕事にする（`根拠台帳` §29）。

主LLM に `say` の欄で名指させる形は、続きの場面の 36.1% しか返さなかった。言い方を強めても
52.8% で、そのぶん偽陽性が増えた。**主LLM にとってこれは五つめの仕事である**（道具を選び、
応答を書き、口調を守り、記憶を申告し、そのうえで判定する）。判定だけをさせる軽量LLM は
36 場面すべてで正しい id を返した。

立ち位置は INSTRUMENT（外から測る）。二つの文が同じ話の続きかは、自分が何を感じたかでは
なく、外からの判断である。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.context_parts import Stance
from familiar_agent.loop.evaluator import Evaluator

_W = (
    "[過去の記憶（証拠つき）: conf<0.55 は不確か]:\n"
    "- 2026-09-07 21:14 id:0123456789ab (適合度:0.81) conf:0.84 (会話): "
    "運動会は8時半に開会式だと伝えた。"
)


def _evaluator(reply: str) -> tuple[Evaluator, MagicMock]:
    utility = MagicMock()
    utility.complete = AsyncMock(return_value=reply)
    context = MagicMock(return_value="[立ち位置]")
    ev = Evaluator(utility, MagicMock(), context=context)
    return ev, context


def test_the_named_id_comes_back():
    ev, _ = _evaluator("0123456789ab")
    assert asyncio.run(ev.judge_follows(_W, "さっきの話だけど")) == "0123456789ab"


def test_none_means_nothing_is_continued():
    ev, _ = _evaluator("none")
    assert asyncio.run(ev.judge_follows(_W, "こんばんは")) is None


def test_a_reply_that_is_not_an_id_is_dropped():
    """12桁でなければ捨てる。形が崩れた返りを、そのまま id として使わない。"""
    for reply in ("わかりません", "0123", "", "id は 0123456789ab です"):
        ev, _ = _evaluator(reply)
        got = asyncio.run(ev.judge_follows(_W, "さっきの話"))
        assert got in (None, "0123456789ab"), reply


def test_an_empty_workspace_asks_nothing():
    """W が空なら、続き先はありえない。呼ばない。"""
    ev, _ = _evaluator("0123456789ab")
    assert asyncio.run(ev.judge_follows("", "こんばんは")) is None
    ev._utility_backend.complete.assert_not_awaited()


def test_the_judge_stands_as_an_instrument():
    """立ち位置は INSTRUMENT。パジュとして感じるのではなく、外から見分ける。"""
    ev, context = _evaluator("none")
    asyncio.run(ev.judge_follows(_W, "こんばんは"))
    assert context.call_args.args[0] is Stance.INSTRUMENT
