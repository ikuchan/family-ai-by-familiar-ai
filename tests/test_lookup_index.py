"""調査の通し番号（設計方針『求めの版チェーン』V1）。

いま検索を識別しているのは**語だけ**である。完了キューの要素は `(語, 結果, 意図 id, 種別)`
で、同じ語で2回投げると区別できない。`_lookup_action_by_query`（語→動作）も語が鍵なので、
重複すると上書きされる。

求めの中の通し番号を振る。求めをまたいだ突き合わせは要らないので、一意な id ではなく
1・2・3… で足りる。同じ語を2回投げること自体は別に止める（`test_no_duplicate_lookup`）。版の content に「1番：search『…』／2番：fetch『…』を起動中」と列挙し、
届いた完了を番号で対応づけるための土台である。

ここでは (1) 投げるたびに番号が増えること、(2) 求めが変わると振り直すこと、(3) 完了キューが
番号を運ぶこと、(4) 並行する別々の調査に別の番号が付くこと、を見る。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backend import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent, _turn


def test_index_starts_at_one_and_increments() -> None:
    """通し番号は1から始まり、投げるたびに増える。"""
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    assert ip._next_lookup_index() == 1
    assert ip._next_lookup_index() == 2
    assert ip._next_lookup_index() == 3


def test_index_resets_per_request() -> None:
    """求めが変われば振り直す（求めをまたいだ突き合わせは要らない）。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
        _turn([ToolCall(id="t2", name="say", input={"text": "はい"})]),
    ])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("ひとつめ")
        first = ip._next_lookup_index()
        await ip.run_iteration("ふたつめ")
        second = ip._next_lookup_index()
        await ip.close()
        return first, second

    first, second = asyncio.run(scenario())
    assert first == 1, "1つめの求めで1から始まっていない"
    assert second == 1, "求めが変わったのに振り直していない"


def test_completion_queue_carries_the_index() -> None:
    """完了キューの要素が通し番号を運ぶ。

    これが無いと、届いた完了がどの調査のものか、語でしか照合できない。
    """
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip.push_completion("さっかー", "結果", index=2)
    item = ip._completion_queue.get_nowait()
    assert item[4] == 2, f"通し番号が運ばれていない: {item}"


def test_distinct_queries_get_distinct_indexes() -> None:
    """違う語には別の番号が付く。

    同じ語を2回投げること自体は `_dispatch_lookup` が止めるので（この求めで一度調べた
    語は二度と調べない・`test_no_duplicate_lookup`）、番号が要るのは並行する別々の調査を
    見分けるためである。
    """
    async def scenario():
        a = _agent(stream_returns=[])

        async def _never_returns(*_a, **_kw):
            await asyncio.sleep(3600)

        a._memory_tool.call = _never_returns
        ip = InformationProcessing(a)
        ip._dispatch_lookup("recall", {"query": "ひとつめ"}, "ひとつめ", None)
        ip._dispatch_lookup("recall", {"query": "ふたつめ"}, "ふたつめ", None)
        got = [(q, idx) for _act, q, idx in ip._in_flight_lookups]
        await ip.close()
        for t in list(ip._tasks):
            t.cancel()
        return got

    assert asyncio.run(scenario()) == [("ひとつめ", 1), ("ふたつめ", 2)]

def test_deferred_completion_gets_its_index_from_the_query() -> None:
    """deferred の完了は語で届くので、語から通し番号を引く。

    `deferred_search` / `deferred_fetch` は `sink(query, result)` と2引数で呼ぶので、
    番号を知らない。同じ語はこの求めで二度投げないので、語からの引き当ては一意になる。
    """
    async def scenario():
        a = _agent(stream_returns=[])

        async def _never_returns(*_a, **_kw):
            await asyncio.sleep(3600)

        a._memory_tool.call = _never_returns
        ip = InformationProcessing(a)
        ip._dispatch_lookup("recall", {"query": "いちばんめ"}, "いちばんめ", None)
        ip._dispatch_lookup("recall", {"query": "にばんめ"}, "にばんめ", None)
        while not ip._completion_queue.empty():
            ip._completion_queue.get_nowait()
        # deferred と同じく、語と結果だけで積む。
        ip.push_completion("にばんめ", "結果")
        item = ip._completion_queue.get_nowait()
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return item

    item = asyncio.run(scenario())
    assert item[4] == 2, f"語から通し番号を引けていない: {item}"
