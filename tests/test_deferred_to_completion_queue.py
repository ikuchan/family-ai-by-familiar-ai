"""deferred（投げっぱなしの外部呼び出し）の完了を、完了キューへ渡す。

正本③：「待たない（投げっぱなし）のは deferred 外部呼び出しだけで、その結果が
**完了キュー→O 経由で次反復の入力**になる」。現状は3つの入口（CUI・GUI・TUI）が
`should_deliver_deferred_result()` を毎周回で問い合わせるポーリングで、キューを介して
いなかった。`EVENT_LOOP` on のときだけキューへ渡し、off では従来どおり溜める（排他）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.tools.deferred_search import DeferredSearchTool


def _tool(sink=None) -> DeferredSearchTool:
    t = DeferredSearchTool(AsyncMock(return_value=("検索の結果", None)), MagicMock())
    if sink is not None:
        t.set_completion_sink(sink)
    return t


def test_completion_goes_to_the_sink_when_wired():
    got: list[tuple[str, str]] = []
    t = _tool(sink=lambda query, result: got.append((query, result)))
    asyncio.run(t._run("明日の天気", "web_search", "user"))
    assert got == [("明日の天気", "検索の結果")]


def test_completion_still_pends_when_no_sink():
    # 旧経路（EVENT_LOOP off）では従来どおり溜める。二重配信にしない。
    t = _tool()
    asyncio.run(t._run("明日の天気", "web_search", "user"))
    assert t.has_pending


def test_sink_replaces_pending_not_duplicates():
    got: list[tuple[str, str]] = []
    t = _tool(sink=lambda query, result: got.append((query, result)))
    asyncio.run(t._run("明日の天気", "web_search", "user"))
    assert len(got) == 1
    assert not t.has_pending      # キューへ渡したぶんは溜めない（二重配信の防止）
