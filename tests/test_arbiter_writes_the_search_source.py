"""調停が調べに行くとき、軽量LLM が検索の担い手（source）も書く（知-am 段 3・2026-10-07）。

軽量LLM が書くのは検索の言葉（query）だけで担い手を書けず、調停が投げる検索はいつも既定の Brave だった（22:04・22:06 の
天気は「リンクの一覧しか出てこない」）。search_deferred のときは source も書かせる。
（出-ay 段 5e で古い分岐の問いの試験 `test_arbiter_judges_with_jev` を外したとき、この 2 件をここへ移した。）
"""

from __future__ import annotations

import pytest

from familiar_agent.loop.arbiter import ArbiterInput


def _inp() -> ArbiterInput:
    return ArbiterInput(utterance="パジュ、明日の天気は？", workspace_ctx="", origin="発話")


def test_the_writer_is_asked_for_the_source_with_a_search():
    from familiar_agent.loop.arbiter import _FIELD_TEXT, _writer_needs

    needs = _writer_needs(_inp(), {"branch": "action", "action": "search_deferred"})
    assert needs[:2] == ["query", "source"]
    assert "tavily" in _FIELD_TEXT["source"] and "brave" in _FIELD_TEXT["source"]
    assert "数字" in _FIELD_TEXT["source"]
    assert "source" not in _writer_needs(_inp(), {"branch": "action", "action": "recall"})


@pytest.mark.parametrize(
    "source,want",
    [
        ("tavily", {"query": "守谷市 明日の天気", "source": "tavily"}),
        ("Brave", {"query": "守谷市 明日の天気", "source": "brave"}),
        ("", None),
        ("google", None),
    ],
)
def test_the_source_reaches_the_tool_input(source, want):
    from familiar_agent.loop.arbiter import assemble

    got = assemble(
        {
            "branch": "action",
            "action": "search_deferred",
            "query": "守谷市 明日の天気",
            "source": source,
        },
        can_see=False,
        origin="発話",
    )
    assert got is not None and got.action == "search_deferred"
    assert (got.tool_input or None) == want
    assert got.query == "守谷市 明日の天気"
