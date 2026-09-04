"""求めを1本の版チェーンにする（設計方針『求めの版チェーン』V3）。

`superseded_by` が2つの意味を兼ねていた。用語一覧は「版履歴専用。**解決には使わない**」と
定めるが、`close_with_children` は求めが決着したときに親（人の発話）と全子（意図 O・完了 O）
へ、まとめ役の id を `superseded_by` として書いていた。意図 O はまとめの旧版ではないし、
人の発話に至っては話者すら違う。

求めを1本の版チェーンにする。各版が求めの状態で、新しい版が前の版を supersede する。これは
版履歴そのものなので、定義と一致する。

- 版1＝求めが立った時点。版2 以降＝調査の起動と結果の到着ごとに1版
- 並行する調査は通し番号で content に列挙し、**鎖は分岐させない**
- 打ち切りも版のひとつ
- 発話の記録と、自分が答えた記録は**鎖の外**
"""

from __future__ import annotations

from familiar_agent.backends import ToolCall

from tests.test_event_loop import _agent, _run_chain, _turn


def _versions(a) -> list:
    """書かれた版の記録（direction="求め"）を、書かれた順に返す。"""
    return [c for c in a._memory.save_async_with_id.call_args_list
            if c.kwargs.get("direction") == "求め"]


def _supersedes(a) -> list[tuple]:
    return [c.args for c in a._memory.mark_superseded.call_args_list]


def test_a_version_is_written_when_a_lookup_starts() -> None:
    """調査を起動すると版が書かれ、前の版を畳む。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")

    versions = _versions(a)
    assert len(versions) >= 2, f"版が2つ以上書かれていない: {len(versions)}"
    assert "1番" in str(versions[0].args[0]), "起動した調査が版に入っていない"


def test_each_version_supersedes_the_previous_one() -> None:
    """版チェーンは1本。新しい版が直前の版だけを畳む。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")

    calls = _supersedes(a)
    # 直前の版だけを畳む（親子のファンアウトではない）。
    for old, _new in calls:
        assert old not in ("obs1",), "発話の記録を畳んでいる（鎖の外のはず）"
    assert calls, "版が前の版を畳んでいない"


def test_the_utterance_record_is_outside_the_chain() -> None:
    """人の発話の記録は鎖の外。畳まれない。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れ"})]),
    ])
    _run_chain(a, utterance="天気は？")
    folded = {old for old, _new in _supersedes(a)}
    assert "obs1" not in folded, "発話の記録が畳まれている"


def test_close_with_children_is_not_used() -> None:
    """`close_with_children` は使わない（親子のファンアウトではなく1本の鎖）。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れ"})]),
    ])
    _run_chain(a, utterance="天気は？")
    assert not a._memory.close_with_children.called, "close_with_children を呼んでいる"


def test_the_version_carries_the_request_and_the_result() -> None:
    """版の content に、求めそのものと届いた結果が入る。"""
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")

    bodies = [str(c.args[0]) for c in _versions(a)]
    assert any("昨日の天気覚えてる？" in b for b in bodies), "求めが版に入っていない"
    assert any("結果が届いた" in b for b in bodies), "結果が版に入っていない"


def test_abort_writes_a_version() -> None:
    """打ち切りも版のひとつとして書く。

    調べかけの途中で話しかけられた場合を作る。求めが閉じたあとでは畳む対象が無いので、
    調査を飛行中のまま `_abort_investigation` を呼ぶ。
    """
    import asyncio

    from familiar_agent.loop.event_loop import InformationProcessing

    async def scenario():
        a = _agent(stream_returns=[])

        async def _never_returns(*_a, **_kw):
            await asyncio.sleep(3600)

        a._memory_tool.call = _never_returns
        ip = InformationProcessing(a)
        ip._origin_text = "天気は？"
        ip._parent_id = "obs1"
        ip._dispatch_lookup("recall", {"query": "天気"}, "天気", None)
        await ip._abort_investigation()
        for t in list(ip._tasks):
            t.cancel()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    bodies = [str(c.args[0]) for c in _versions(a)]
    assert any("打ち切った" in b for b in bodies), (
        f"打ち切りが版として残っていない: {bodies}"
    )
    assert any("天気は？" in b for b in bodies), "求めが版に入っていない"
