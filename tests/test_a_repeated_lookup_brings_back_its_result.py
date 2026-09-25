"""同じ語を二度投げないとき、**前に返ったものをそのまま返す**（出-ag-ろ 穴 3・2026-09-25）。

実機 2026-09-21 17:34：「3分測って」→ `set_timer` は「まだ掛けていない。本人に一度聞く：「…」」を
返した。調停が同じ語でもう一度 `set_timer` を選び、守りは「この求めですでに調べた。**結果は W に
ある**」と返した。ところがタイマーの返りの反復は想起をしないので、**W に結果は無かった**。
材料の無い W を見て、調停は `light` で「タイマーをセットしました」と言った（掛かっていない）。

返り文が事実と違っていた。前に返ったものを載せれば、確かめの問いがもう一度 W に載る。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

ASKED = "まだ掛けていない。本人に一度聞く：「3 分のタイマーね。その間は黙って聞かないよ、いい？」"


def _second_try(first_result: "str | None") -> str:
    """同じ語を 2 回投げ、2 回目に積まれた完了の文を返す。"""

    async def scenario():
        a = _agent(stream_returns=[])

        async def _never_returns(*_a, **_kw):
            await asyncio.sleep(3600)

        a._memory_tool.call = _never_returns
        ip = InformationProcessing(a)
        ip._dispatch_lookup("recall", {"query": "タイマー"}, "タイマー", None)
        ip._req.lookups[0].result = first_result  # None なら、まだ返っていない
        while not ip._triggers.empty():
            ip._triggers.get_nowait()
        ip._dispatch_lookup("recall", {"query": "タイマー"}, "タイマー", None)
        got = ip._triggers.get_nowait().result
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return got

    return asyncio.run(scenario())


def test_the_earlier_result_comes_back():
    got = _second_try(ASKED)
    assert ASKED in got, got
    assert "すでに調べた" in got


def test_it_does_not_point_at_a_w_that_may_not_hold_it():
    """「結果は W にある」と言わない。タイマーの返りの反復は想起をせず、W に載らない。"""
    assert "W にある" not in _second_try(ASKED)


def test_a_query_still_in_flight_says_so():
    got = _second_try(None)
    assert "まだ返っていない" in got, got
    assert "W にある" not in got
