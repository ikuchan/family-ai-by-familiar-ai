"""道具の定義もキャッシュに載せる（出-i）。

§27 で確定した「`claude-haiku-4-5` ＋キャッシュ」の**本体**である。安定部だけ（3,342
トークン）では Haiku 系の最小長 4,096 に届かず**キャッシュが1回も効かない**。道具の定義
（2,329トークン）を載せると跨ぎ、**安定部ごと全部が乗る**（実測：読み 5,348・生 405）。
1000ターン1反復で 738円 → 366円 になる。

**印は最後の道具に付ける。** Anthropic の `cache_control` は「ここまで」を意味する境目で、
道具の並びの最後に付けると**道具全体＋その前の安定部**が範囲に入る。

**モデルが変わると最適な構成が変わる。** `sonnet-5` は最小長が低く安定部だけで効くので、
道具を載せると読み出し料が増えて**かえって高くなる**（581円 → 635円）。だから
`cache_tools` は指定できる形にし、既定は「載せる」（いまの主LLM は `haiku`）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _loop():
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = object.__new__(InformationProcessing)
    ip._agent = MagicMock()
    return ip


def _fake_defs(*names: str) -> list[dict]:
    return [{"name": n, "description": n, "input_schema": {"type": "object"}} for n in names]


def test_the_last_tool_carries_the_cache_mark():
    """境目は最後の1つに付く。**道具全体とその前の安定部**が範囲に入る。"""
    ip = _loop()
    ip._ACTIONS = {"say": lambda a: _fake_defs("say"),
                   "recall": lambda a: _fake_defs("recall")}
    defs = ip._tools(actions=("say", "recall"))
    assert len(defs) == 2
    assert "cache_control" not in defs[0]
    assert defs[-1]["cache_control"] == {"type": "ephemeral"}


def test_turning_it_off_leaves_every_tool_bare():
    """`sonnet` のように安定部だけで効くモデルでは、載せるとかえって高い。"""
    ip = _loop()
    ip._ACTIONS = {"say": lambda a: _fake_defs("say")}
    defs = ip._tools(actions=("say",), cache_tools=False)
    assert all("cache_control" not in d for d in defs)


def test_no_tools_means_no_mark():
    ip = _loop()
    ip._ACTIONS = {}
    assert ip._tools(actions=("say",)) == []


def test_the_mark_does_not_leak_into_the_shared_definition():
    """`get_tool_definitions()` が返す実体を書き換えない。

    書き換えると、次に取ったときも印が付いたままになり、`cache_tools=False` が効かなく
    なる（**同じ辞書を使い回している**）。
    """
    ip = _loop()
    shared = _fake_defs("say")
    ip._ACTIONS = {"say": lambda a: shared}
    ip._tools(actions=("say",))
    assert "cache_control" not in shared[0]


@pytest.mark.parametrize("n", [1, 2, 5])
def test_only_one_mark_no_matter_how_many_tools(n):
    """境目は1つ。複数付けるとキャッシュの区切りが増えて、書き込みが増える。"""
    ip = _loop()
    names = tuple(f"t{i}" for i in range(n))
    ip._ACTIONS = {nm: (lambda a, _nm=nm: _fake_defs(_nm)) for nm in names}
    defs = ip._tools(actions=names)
    assert sum("cache_control" in d for d in defs) == 1
