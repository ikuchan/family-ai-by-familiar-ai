"""調べかけの途中に話しかけられたら、その調査を打ち切る。

人が言い直したとき、前の調査を続ける意味はない。実機で「これはどこの地方の天気？」に
答えられず、言い直されたあとも同じ検索を繰り返した（同じ語・同じ結果を4反復）。

**結果は捨てる**（行き先の親が閉じるので、残すと次の求めの W に無関係な完了が載る）。
ただし**何を打ち切ったかは記録に残す**（あとで「あのとき何を調べていたか」を辿れる）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backend import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _turn


def _ip_with_investigation():
    a = _agent(stream_returns=[
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    ip = InformationProcessing(a)
    ip._parent_id = "obs-parent"
    ip._chain_head_id = "obs-child"
    ip._in_flight_lookups = [("search_deferred", "明日の天気")]
    ip._lookup_action_by_query = {"明日の天気": "search_deferred"}
    ip._inflight = 1
    ip._completion_queue.put_nowait(("明日の天気", "晴れ", "obs-child"))
    return a, ip


def test_pending_completions_are_dropped():
    a, ip = _ip_with_investigation()
    asyncio.run(ip._abort_investigation())
    assert ip._completion_queue.empty()
    assert ip._inflight == 0
    assert ip._in_flight_lookups == []
    assert ip._lookup_action_by_query == {}


def test_what_was_dropped_is_recorded():
    a, ip = _ip_with_investigation()
    asyncio.run(ip._abort_investigation())
    written = [c for c in a._memory.save_async_with_id.call_args_list
               if c.kwargs.get("direction") == "中断"]
    assert len(written) == 1
    assert "search_deferred「明日の天気」" in written[0].args[0]
    assert written[0].kwargs["parent_id"] == "obs-parent"


def test_the_record_closes_the_abandoned_chain():
    # 打ち切りの記録で親と生きた子を閉じる（鎖は1件へ収束する）。
    a, ip = _ip_with_investigation()
    asyncio.run(ip._abort_investigation())
    parent, new_id = a._memory.close_with_children.call_args.args
    assert parent == "obs-parent"
    assert new_id != parent
    assert ip._parent_id is None and ip._chain_head_id is None


def test_nothing_happens_when_there_was_no_investigation():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    asyncio.run(ip._abort_investigation())
    a._memory.save_async_with_id.assert_not_awaited()
