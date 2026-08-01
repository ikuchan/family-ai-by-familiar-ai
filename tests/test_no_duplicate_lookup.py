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

from familiar_agent.loop.event_loop import InformationProcessing

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
        got = list(ip._in_flight_lookups)
        for t in list(ip._tasks):
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
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return items

    items = asyncio.run(scenario())
    assert items, "止めたのに完了が積まれていない"
    assert items[0][3] == "完了", f"種別が完了でない: {items[0]}"
    assert "調べた" in items[0][1], f"すでに調べた旨が入っていない: {items[0][1]}"


def test_a_finished_query_is_still_blocked() -> None:
    """結果が届いたあとでも、同じ語は二度と調べない。

    `recall` は DB を引くだけなので、引き直しても結果は変わらない。取り直しに意味がない。
    """
    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "済んだ語"}, "済んだ語", None)
        # 結果が届いて飛行中から外れた状態を作る。
        ip._in_flight_lookups.clear()
        ip._dispatch_lookup("recall", {"query": "済んだ語"}, "済んだ語", None)
        got = list(ip._in_flight_lookups)
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return got

    assert asyncio.run(scenario()) == [], "済んだ語をもう一度調べている"


def test_a_different_query_still_goes_out() -> None:
    """違う語は通る（止めるのは重複だけ）。"""
    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "ひとつめ"}, "ひとつめ", None)
        ip._dispatch_lookup("recall", {"query": "ふたつめ"}, "ふたつめ", None)
        got = [(q, idx) for _act, q, idx in ip._in_flight_lookups]
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return got

    assert asyncio.run(scenario()) == [("ひとつめ", 1), ("ふたつめ", 2)]


def test_a_new_request_clears_the_history() -> None:
    """求めが変われば、同じ語をまた調べられる。"""
    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "天気"}, "天気", None)
        for t in list(ip._tasks):
            t.cancel()
        ip._tasks.clear()
        await ip._abort_investigation()          # 求めの区切り
        ip._dispatch_lookup("recall", {"query": "天気"}, "天気", None)
        got = list(ip._in_flight_lookups)
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return got

    assert len(asyncio.run(scenario())) == 1, "求めが変わったのに調べられない"


def test_a_blocked_lookup_is_counted_as_inflight() -> None:
    """止めた調査も飛行中として数える。

    完了を積む以上、飛行中として数えないと帳尻が合わない。取込は**積まれた完了1件につき
    `_inflight` を1つ減らす**ので、増やさずに積むと実際より小さくなる。飛行中の調査が
    残っているのに 0 になると、駆動体が「調査中ではない」とみなして待ち方を変える。
    """
    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        first = ip._inflight
        ip._dispatch_lookup("recall", {"query": "同じ語"}, "同じ語", None)
        second = ip._inflight
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return first, second

    first, second = asyncio.run(scenario())
    assert first == 1, f"投げた調査が数えられていない: {first}"
    assert second == 2, f"止めた調査が数えられていない: {second}"


def test_inflight_returns_to_zero_after_intake() -> None:
    """積んだぶんを取り込むと、飛行中の数が 0 へ戻る（増減が釣り合う）。"""
    async def scenario():
        a, ip = _ip_with_slow_recall()
        ip._dispatch_lookup("recall", {"query": "語"}, "語", None)
        ip._dispatch_lookup("recall", {"query": "語"}, "語", None)   # 止められる
        ip._in_flight_lookups.clear()          # 1件目の結果が届いた体にする
        ip._completion_queue.put_nowait(("語", "結果", None, "完了", 1))
        await ip._intake()
        got = ip._inflight
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return got

    # 2 増えて 2 減る（止めた分の完了＋届いた分の完了）。
    assert asyncio.run(scenario()) == 0, "飛行中の数が釣り合っていない"
