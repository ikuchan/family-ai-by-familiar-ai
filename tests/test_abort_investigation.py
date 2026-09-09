"""調べかけの途中に話しかけられたら、その調査を打ち切る。

人が言い直したとき、前の調査を続ける意味はない。実機で「これはどこの地方の天気？」に
答えられず、言い直されたあとも同じ検索を繰り返した（同じ語・同じ結果を4反復）。

**結果は捨てる**（行き先の親が閉じるので、残すと次の求めの W に無関係な完了が載る）。
ただし**何を打ち切ったかは記録に残す**（あとで「あのとき何を調べていたか」を辿れる）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Completion
from tests.test_event_loop import _agent, _turn


def _ip_with_investigation():
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
        ]
    )
    ip = InformationProcessing(a)
    ip._req.request_id = "obs-parent"
    ip._lookups = [Lookup(index=1, action="search_deferred", query="明日の天気", generation=0)]
    ip._completion_queue.put_nowait(
        Completion(kind="完了", query="明日の天気", result="晴れ", intent_id="obs-child", index=1)
    )
    return a, ip


def test_pending_completions_are_dropped():
    a, ip = _ip_with_investigation()
    asyncio.run(ip._abort_lookups())
    assert ip._completion_queue.empty()
    assert ip._in_flight_count == 0
    assert ip._lookups == []


def test_nothing_happens_when_there_was_no_investigation():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    asyncio.run(ip._abort_lookups())
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
        ip._req.utterance = "前の問い"
        ip._request_generation = 0
        # 反復の途中で打ち切られた状況を作る（生成が返る前に世代が進む）。
        original = a.backend.stream_turn

        async def _bump(*args, **kwargs):
            ip._request_generation += 1
            return await original(*args, **kwargs)

        a.backend.stream_turn = _bump
        await ip._iterate()
        await ip.close()

    asyncio.run(scenario())
    assert shown == []  # 何も言わない
    a._tts.call.assert_not_awaited()


def test_a_completion_from_an_abandoned_request_is_dropped():
    # 外部呼び出しは投げた時点で飛んでいる。打ち切ったあとに届いても捨てる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._lookups = [Lookup(index=1, action="search_deferred", query="明日の天気", generation=0)]
    ip._request_generation = 1
    ip.push_completion("明日の天気", "晴れ")
    assert ip._completion_queue.empty()


def test_the_abort_is_written_as_a_version():
    """打ち切りは求めの版として書く。

    以前は `direction="中断"` の記録を書き、`close_with_children` で親と生きた子を閉じて
    いた。版チェーンでは打ち切りも求めの状態のひとつなので、1つの版として書き、直前の版
    だけを畳む（親子のファンアウトは無い）。
    """
    a, ip = _ip_with_investigation()
    asyncio.run(ip._abort_lookups())

    versions = [
        c
        for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "求め"
    ]
    assert len(versions) == 1, f"打ち切りの版が1件でない: {len(versions)}"
    body = str(versions[0].args[0])
    assert "打ち切った" in body, f"打ち切りが分からない: {body}"
    assert "search_deferred「明日の天気」" in body, "何を打ち切ったかが残っていない"
    assert versions[0].kwargs["parent_id"] == "obs-parent"
    assert not a._memory.close_with_children.called, "close_with_children を呼んでいる"


def test_the_abort_closes_the_exchange():
    """打ち切りでやりとりを閉じる。

    打ち切りの版の親は、打ち切られた求めの起点である（`_request_id` を捨てる前に書く）。
    ところが、やりとりの項は `_finish` までに控えた並びから作られ、打ち切りは `_finish` を
    通らない。閉じないと並びが次のターンへ持ち越され、**一つのやりとりに起点が2つ**入る
    （聞かれて答える前に話題が変わった分と、新しい問いが混ざる）。

    答えも要約も無いやりとりになるが、それが起きた事実そのものである。
    """
    a, ip = _ip_with_investigation()
    ip._req.turn_records = [("obs1", "起点"), ("obs2", "版")]

    asyncio.run(ip._abort_lookups())

    members = a._memory.record_exchange.call_args.args[0]
    roles = [r for _, r, _ in members]
    assert roles.count("起点") == 1, f"起点が1つでない: {members}"
    assert "答え" not in roles, "答えていないのに答えの項がある"
    assert [i for i, _, _ in members][:2] == ["obs1", "obs2"]
    # 打ち切りの版も同じやりとりに入る（親は同じ起点）。
    assert roles[-1] == "版"


def test_the_carry_over_for_the_diffuse_pool_survives_the_abort():
    """母集合への持ち越しは残す。

    打ち切った調査と、言い直した問いの共起は、たどる価値がある（WR の約束）。やりとりを
    閉じることと、母集合へ持ち越すことは別の規則である。ここが一緒に消えると、打ち切りの
    記録へ辿り着く辺が拡散想起から無くなる。
    """
    a, ip = _ip_with_investigation()
    ip._req.turn_records = [("obs1", "起点"), ("obs2", "版")]

    asyncio.run(ip._abort_lookups())

    assert [i for i, _ in ip._req.turn_records][:2] == ["obs1", "obs2"], "持ち越しまで消している"
    # 次のやりとりは、打ち切りの次から始まる。
    assert ip._req.exchange_start == len(ip._req.turn_records)
