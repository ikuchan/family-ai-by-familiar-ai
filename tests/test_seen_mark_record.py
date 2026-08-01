"""見たことを、求めの鎖の外に独立した記録として書く（知-c の前提）。

実機で見回りが成立しなかった。Drive は発火し（`Drive fired: seeking`）、`see` は動いて
意味づけも通ったが（`出入り口で見えたもの 2 件：boy、man`）、**`look` が一度も選ばれず、
2回とも同じ定点だった**。

原因は、見た印が O に独立した記録として残らないことである。見えたものは求めの版
（`direction="求め"`）の中の一節としてしか存在せず、版は次の版が来るたびに畳まれる。
そのため「どの定点を最後に見てからどれだけ経ったか」の順序が O に作られず、W から
「次はここを見る番だ」という手がかりが出ない。ユースケース③ が定める「巡回は次回以降の
見回りで W が次の薄れた定点を選ぶことで創発する」が成り立たない。

会話はすでに三分割になっている。きっかけ（`direction="発話"`・鎖の外）、途中経過
（`direction="求め"`・畳む）、結果（`direction="発話"` の「自分が答えた」・鎖の外）。
見回りは発話しないので結果の記録が書かれず、そこだけが欠けていた。

**旧 `run()` はこれを `direction="観察"` で書いていた。** 本番 DB に 256 件あり、いずれも
「もっと左を向いたよ！収納棚に…」のような見たものの記録で、2026-07-24 を最後に途絶えて
いる。新しいループへ移ったとき `camera_used=False` が渡るようになり、書き込みが到達
しなくなったためである。同じ意味の記録なので `direction` は分けず、`観察` を使う。

版には見えたものを残さない。残すと同じ出来事が2件になり、想起でどちらも上がって W の枠を
食う。版は「何番の調査が届いたか」だけを持つ。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(origin: str = "周りを見て"):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._origin_text = origin
    return a, ip


def _saved(agent, direction: str) -> list[str]:
    """その direction で書かれた content を、書かれた順に返す。"""
    return [c.args[0] for c in agent._memory.save_async_with_id.call_args_list
            if c.kwargs.get("direction") == direction]


def test_a_seen_result_is_written_as_its_own_record() -> None:
    """見たことは `direction="観察"` の独立した記録になる。

    書き手は `_run_camera`（実際にカメラを回した経路）である。中身の検証は
    `test_seen_mark_content` にある。
    """
    async def scenario():
        a, ip = _ip()
        await ip._write_seen_mark("窓側を見た。見えたもの：椅子、窓")
        await ip.close()
        return a

    a = asyncio.run(scenario())
    got = _saved(a, "観察")
    assert got, "観察の記録が書かれていない"
    assert "窓側を見た" in got[0], f"定点名が入っていない: {got[0]}"
    assert "椅子" in got[0], f"見えたものが入っていない: {got[0]}"


def test_the_seen_record_is_not_folded() -> None:
    """見た印は畳まない（畳むと想起の候補から外れ、薄れの順序が作れない）。"""
    async def scenario():
        a, ip = _ip()
        await ip._write_seen_mark("襖側を見た。見えたもの：戸")
        # さらに版が進んでも、観察の記録は畳まれない。
        ip._completion_queue.put_nowait(("語", "結果", None, "完了", 2))
        await ip._intake()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    folded = [c.args for c in a._memory.mark_superseded.call_args_list]
    seen_ids = [c for c in folded if "観察" in str(c)]
    assert not seen_ids, f"観察の記録を畳んでいる: {folded}"


def test_the_version_does_not_carry_what_was_seen() -> None:
    """版に見えたものを載せない（同じ出来事を2件にしない）。"""
    async def scenario():
        a, ip = _ip()
        ip._lookup_action_by_query["目の前を見る"] = "see"
        ip._completion_queue.put_nowait(
            ("目の前を見る", "窓側を見た。見えたもの：椅子、窓", None, "完了", 1))
        await ip._intake()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    versions = _saved(a, "求め")
    assert versions, "版が書かれていない"
    assert "椅子" not in versions[-1], f"版に見えたものが載っている: {versions[-1]}"


def test_the_version_still_says_the_lookup_finished() -> None:
    """版には「何番が届いたか」は残す（求めの状態は分かる必要がある）。"""
    async def scenario():
        a, ip = _ip()
        ip._lookup_action_by_query["目の前を見る"] = "see"
        ip._completion_queue.put_nowait(
            ("目の前を見る", "窓側を見た。見えたもの：椅子", None, "完了", 1))
        await ip._intake()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    last = _saved(a, "求め")[-1]
    assert "1番" in last, f"通し番号が落ちている: {last}"
    assert "see" in last, f"どの動作か落ちている: {last}"


def test_other_lookups_still_carry_their_result_in_the_version() -> None:
    """`recall` や検索の結果は従来どおり版に載る（分けるのは `see` だけ）。"""
    async def scenario():
        a, ip = _ip("昨日の天気は？")
        ip._lookup_action_by_query["昨日 天気"] = "search_deferred"
        ip._completion_queue.put_nowait(
            ("昨日 天気", "西日本は晴れだった", None, "完了", 1))
        await ip._intake()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    last = _saved(a, "求め")[-1]
    assert "西日本は晴れだった" in last, f"結果が版から消えている: {last}"
    assert not _saved(a, "観察"), "検索の結果まで観察として書いている"


def test_the_seen_record_reaches_the_workspace() -> None:
    """観察の記録が W へ載る。

    版から見えたものを落とすので、この経路が無いと `see` した反復の次で、調停が
    何が見えたかを知らないまま返事を作る。
    """
    async def scenario():
        a, ip = _ip()
        ip._lookup_action_by_query["目の前を見る"] = "see"
        ip._completion_queue.put_nowait(
            ("目の前を見る", "窓側を見た。見えたもの：椅子", None, "完了", 1))
        await ip._intake()
        got = list(ip._wr_ids)
        await ip.close()
        return a, got

    a, wr = asyncio.run(scenario())
    assert wr, "W へ何も渡していない"
