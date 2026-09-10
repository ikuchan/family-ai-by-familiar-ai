"""**会話入力も待ち行列を通る**（環-f-い-2）。

きっかけは4つ——会話入力・知覚イベント・情動発火・完了。3つは列に並んでいたが、人の言葉
だけが `begin_request()` の同期呼び出しで横から入り、反復が呼び手のタスクの上で回っていた。
列へ移すと**順序づけが `_take_trigger()` の1箇所**に集まる。

**種別に `発話` を使わない。** その語はコードの中で3つの別物を指している（人が話しかけた
こと・パジュがつないだ一言・パジュが答えたこと）。用語一覧の語は `会話入力` である。

**呼び手の約束は変わらない。** `push_utterance()` は「その反復の出力」を返す——`begin_request`
だったころも返していたのは求めの終わりではなく最初の反復の出力で、意味は同じである。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Trigger


def _ip():
    ip = InformationProcessing(MagicMock())
    ip._abort_lookups = AsyncMock()
    ip._begin_request = AsyncMock()
    return ip


def test_an_utterance_is_queued_not_run_on_the_caller():
    """人の言葉は列へ積まれ、**反復は駆動体の上で回る**。"""
    ip = _ip()
    ran_on: list[str] = []

    async def _iterate():
        ran_on.append(asyncio.current_task().get_name())
        return "はい"

    ip._iterate = _iterate

    async def scenario():
        ip._ensure_driver()
        caller = asyncio.current_task().get_name()
        spoken = await ip.push_utterance("こんにちは")
        await ip.close()
        return spoken, caller

    spoken, caller = asyncio.run(scenario())
    assert spoken == "はい"
    assert ran_on and ran_on[0] != caller, "呼び手のタスクの上で回っている"


def test_an_utterance_is_taken_before_a_device_or_an_affect():
    """**人の言葉が最優先。** 機器・情動より先に採る。

    優先順位は「**同時に届いたものの中から1つ選ぶとき**」の規則である。選ばれなかった
    ものは保留箱で待ち、**待った順に**片づく（待たせたのだから、待った順で出す）。
    """
    ip = _ip()

    async def scenario():
        ip.push_affect("SEEKING", "気になる")  # 先に届いた
        ip.push_device("入室", "パパ が来た")
        ip._triggers.put_nowait(Trigger(kind="会話入力", query="ねえ"))
        return [(await ip._take_trigger()).kind for _ in range(3)]

    assert asyncio.run(scenario()) == ["会話入力", "情動", "機器"]


def test_an_utterance_is_never_held_while_a_lookup_is_in_flight():
    """調査中でも待たせない。人が言い直したら前の調査は打ち切る側である。"""
    from familiar_agent.loop.request import Lookup

    ip = _ip()

    async def scenario():
        ip._req.lookups = [Lookup(index=1, action="recall", query="q", generation=0)]
        ip._triggers.put_nowait(Trigger(kind="会話入力", query="ねえ"))
        ip.push_affect("SEEKING", "気になる")
        got = await ip._take_trigger()
        return got.kind, [t.kind for t in ip._held]

    kind, held = asyncio.run(scenario())
    assert kind == "会話入力"
    assert held == ["情動"]  # 情動だけが待たされる


def test_cancelling_the_waiter_stops_the_iteration():
    """**停止ボタンが効く。** GUI は呼び手のタスクを cancel する形で中断する。反復は
    駆動体の上で回るので、待ち手の cancel を受けてこちらから止める。"""
    ip = _ip()
    finished = False

    async def _iterate():
        nonlocal finished
        await asyncio.sleep(1.0)
        finished = True
        return "遅い返事"

    ip._iterate = _iterate

    async def scenario():
        ip._ensure_driver()
        task = asyncio.create_task(ip.push_utterance("ねえ"))
        await asyncio.sleep(0.05)  # 駆動体が拾って反復を始めた頃
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)
        alive = ip._driver is not None and not ip._driver.done()
        await ip.close()
        return alive

    alive = asyncio.run(scenario())
    assert not finished, "中断したのに反復が最後まで走った"
    assert alive, "駆動体まで畳んでしまった（次のきっかけで起きなくなる）"


def test_closing_does_not_leave_a_caller_waiting():
    """終了時に、列や保留箱へ残った会話入力の待ち手を起こす（`await` が永久に返らない）。"""
    ip = _ip()

    async def scenario():
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        ip._triggers.put_nowait(Trigger(kind="会話入力", query="ねえ", future=fut))
        await ip.close()
        return fut.cancelled()

    assert asyncio.run(scenario())


def test_the_caller_wakes_before_the_loop_moves_on():
    """**待っている呼び手を、次のきっかけより先に起こす。**

    待ち行列が空でなければ `Queue.get()` は譲らずに返る。駆動体が譲らなければ、そのまま
    次の反復まで進んでから呼び手の `await` が再開する。`begin_request` だったころは反復が
    呼び手のタスクの上で回っており、戻った時点は求めの続きより前だった。
    """
    ip = _ip()
    order: list[str] = []

    async def _iterate():
        order.append("反復")
        # この反復のあとに、続きのきっかけが列へ積まれている状況を作る。
        if len(order) == 1:
            ip._triggers.put_nowait(Trigger(kind="完了", query="q", result="r"))
        return "はい"

    ip._iterate = _iterate

    async def scenario():
        ip._ensure_driver()
        await ip.push_utterance("ねえ")
        order.append("呼び手が再開")
        await ip.close()

    asyncio.run(scenario())
    assert order[:2] == ["反復", "呼び手が再開"], order
