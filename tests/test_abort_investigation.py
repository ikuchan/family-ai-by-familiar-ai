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
    ip._completion_queue.put_nowait(("明日の天気", "晴れ", "obs-child", "完了"))
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


def test_a_running_iteration_is_folded_after_an_abort():
    """打ち切られたら、走っている反復は出力せずに畳む。

    打ち切りの時点で外部呼び出しは既に飛んでおり、反復もフルLLM の返りを待っている
    最中なので、止めるには世代番号で見分けるしかない。実機では、打ち切った直後に
    走っていた反復が `fetch_deferred` を投げ、返事も1つ余計に出た（「そうですか」が2回）。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "余計な返事"})])])
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        ip.set_output(shown.append)
        ip._utterance = "前の問い"
        ip._generation = 0
        # 反復の途中で打ち切られた状況を作る（生成が返る前に世代が進む）。
        original = a.backend.stream_turn

        async def _bump(*args, **kwargs):
            ip._generation += 1
            return await original(*args, **kwargs)

        a.backend.stream_turn = _bump
        await ip._iterate()
        await ip.close()

    asyncio.run(scenario())
    assert shown == []                       # 何も言わない
    a._tts.call.assert_not_awaited()


def test_a_completion_from_an_abandoned_request_is_dropped():
    # 外部呼び出しは投げた時点で飛んでいる。打ち切ったあとに届いても捨てる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._lookup_generation["明日の天気"] = 0
    ip._generation = 1
    ip.push_completion("明日の天気", "晴れ")
    assert ip._completion_queue.empty()
