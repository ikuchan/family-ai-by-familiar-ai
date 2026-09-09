"""同じ語は、この求めのあいだ二度と調べない（V1 の積み残し）。

`deferred_search` と `deferred_fetch` は、同じ意図の調査が飛行中・保留中なら投げない
（軽量LLM が言い換えまで見て判定する）。ところが `recall`・`see`・`look` は素通りで、
実機では**同じ `recall` を4反復続けて投げた**（語は MD5 まで一致）。4回とも同じ記憶を
取ってきて、4反復ぶん無駄になった。

`recall` は指定した語で自分と在席者の記憶を横断して探すツールで、DB を引くだけである。
同じ語で引き直しても結果は変わらないので、取り直しに意味がない。

止めた調査は「すでに調べた」旨を完了として積み、反復を続ける（deferred と同じ形）。
投げずに黙って帰ると、完了も時間切れも来ないまま飛行中の数だけが残り、駆動体が待ち
続ける。

これで「1つの求めのあいだ、語は重複しない」が全動作で成り立ち、届いた完了を語から
通し番号へ引き当てられる。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing, Completion

from tests.test_event_loop import _agent


def _ip_with_slow_recall():
    """完了が届かないようにして、飛行中のまま観察する。"""
    a = _agent(stream_returns=[])

    async def _never_returns(*_a, **_kw):
        await asyncio.sleep(3600)

    a._memory_tool.call = _never_returns
    return a, InformationProcessing(a)


def test_the_same_query_is_not_dispatched_twice() -> None:
    """同じ語を2回投げようとしても、2回目は投げない。"""

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        got = [(lk.action, lk.query, lk.index) for lk in ip._lookups]
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return got

    assert len(asyncio.run(scenario())) == 1, "同じ語で2件飛んでいる"


def test_a_blocked_lookup_is_pushed_as_a_completion() -> None:
    """止めた調査は完了として積む（駆動体が待ち続けないため）。"""

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        while not ip._completion_queue.empty():
            ip._completion_queue.get_nowait()
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        items = []
        while not ip._completion_queue.empty():
            items.append(ip._completion_queue.get_nowait())
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return items

    items = asyncio.run(scenario())
    assert items, "止めたのに完了が積まれていない"
    assert items[0].kind == "完了", f"種別が完了でない: {items[0]}"
    assert "調べた" in items[0].result, f"すでに調べた旨が入っていない: {items[0].result}"


def test_a_finished_query_is_still_blocked() -> None:
    """結果が届いたあとでも、同じ語は二度と調べない。

    `recall` は DB を引くだけなので、引き直しても結果は変わらない。取り直しに意味がない。
    """

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "済んだ語"}, "済んだ語", None)
        # 結果が届いて飛行中から外れた状態を作る。
        ip._lookups.clear()
        ip._dispatch_lookup("recall", {"query": "済んだ語"}, "済んだ語", None)
        got = [(lk.action, lk.query, lk.index) for lk in ip._lookups]
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return got

    # 器は1件のまま。二度目は投げられず、新しい器も作られない。
    assert asyncio.run(scenario()) == [("recall", "済んだ語", 1)], "済んだ語をもう一度調べている"


def test_a_different_query_still_goes_out() -> None:
    """違う語は通る（止めるのは重複だけ）。"""

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "ひとつめ"}, "ひとつめ", None)
        ip._dispatch_lookup("recall", {"query": "ふたつめ"}, "ふたつめ", None)
        got = [(lk.query, lk.index) for lk in ip._lookups]
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return got

    assert asyncio.run(scenario()) == [("ひとつめ", 1), ("ふたつめ", 2)]


def test_a_new_request_clears_the_history() -> None:
    """求めが変われば、同じ語をまた調べられる。"""

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "天気"}, "天気", None)
        for t in list(ip._background_tasks):
            t.cancel()
        ip._background_tasks.clear()
        await ip._abort_lookups()  # 求めの区切り
        ip._dispatch_lookup("recall", {"query": "天気"}, "天気", None)
        got = [(lk.action, lk.query, lk.index) for lk in ip._lookups]
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return got

    assert len(asyncio.run(scenario())) == 1, "求めが変わったのに調べられない"


def test_a_blocked_lookup_does_not_add_a_second_record() -> None:
    """止めた調査で器を増やさない（環-g・段は で挙動が変わった）。

    以前は「飛行中の数」を手で持っており、止めた調査でも数だけ増やし、取込が減らすことで
    帳尻を合わせていた。**いまは数が器から導かれる**ので、増やす必要がない。器を2つ作ると、
    同じ語で引いたときどちらが返るか決まらなくなる。
    """

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        first = (ip._in_flight_count, len(ip._lookups))
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        second = (ip._in_flight_count, len(ip._lookups))
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return first, second

    first, second = asyncio.run(scenario())
    assert first == (1, 1), f"投げた調査が器に無い: {first}"
    assert second == (1, 1), f"止めた調査で器が増えている: {second}"


def test_the_count_returns_to_zero_when_the_result_arrives() -> None:
    """結果が届けば飛行中でなくなる。**手で減らさない**（数は導出）。"""

    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "語"}, "語", None)
        ip._dispatch_lookup("recall", {"query": "語"}, "語", None)  # 止められる
        before = ip._in_flight_count
        ip._completion_queue.put_nowait(Completion(kind="完了", query="語", result="結果", index=1))
        await ip._intake()
        after = ip._in_flight_count
        for t in list(ip._background_tasks):
            t.cancel()
        await ip.close()
        return before, after

    before, after = asyncio.run(scenario())
    assert before == 1
    assert after == 0, f"結果が届いたのに飛行中のまま: {after}"
