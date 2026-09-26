"""打ち切られた求めの後始末は、次の求めの状態に触らない（出-au 段 1-5・2026-09-26・`設計方針_判定の段` §2.1）。

求めの状態（`self._req`）は 1 つを共有している。返事を声にしている最中（再生を待っている間）に名前で呼ばれて
打ち切りが入ると、`_abort_lookups` が状態を空にし、次の求めが始まる。そのあとで前の反復が `_finish` に着くと、
**次の求めの id を親にして記録を書き、次の求めの状態を空にし、やりとりを閉じていた**。

打ち切られた求め（世代が進んでいる）の `_finish` は、声になった事実だけを書き（親は付けない）、ほかは何もしない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    a._oif.write = AsyncMock(return_value="obs-answer")
    a._spawn_background_task = MagicMock()
    ip = InformationProcessing(a)
    ip._request_generation = 1  # 打ち切りで世代が進んだ
    ip._req.request_id = "req-next"  # 次の求めがもう始まっている
    ip._req.told_unsaid = ["obs-unsaid"]
    ip._fold_told = AsyncMock()
    ip._close_exchange = MagicMock(return_value=[])
    ip._notify_request_state = MagicMock()
    return ip, a


def test_a_spoken_reply_of_an_aborted_request_is_recorded_without_touching_the_next():
    ip, a = _ip()
    asyncio.run(ip._finish("晴れだよ", [], "発話", gen=0))
    a._oif.write.assert_awaited_once()
    mi = a._oif.write.await_args.args[0]
    assert mi.content == "自分が答えた：晴れだよ" and mi.parent_id is None
    assert ip._req.request_id == "req-next"  # 次の求めのまま
    assert ip._req.told_unsaid == ["obs-unsaid"]
    ip._fold_told.assert_not_awaited()
    ip._close_exchange.assert_not_called()
    ip._notify_request_state.assert_not_called()
    a._spawn_background_task.assert_not_called()


def test_an_unspoken_reply_of_an_aborted_request_leaves_nothing():
    ip, a = _ip()
    asyncio.run(ip._finish("晴れだよ", [], "独白", gen=0))
    a._oif.write.assert_not_awaited()
    assert ip._req.request_id == "req-next"


def test_the_current_generation_still_finishes_normally():
    ip, a = _ip()
    asyncio.run(ip._finish("晴れだよ", [], "発話", gen=1))
    assert ip._req.request_id is None
    ip._notify_request_state.assert_called_once_with(False)
