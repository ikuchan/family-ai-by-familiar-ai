"""完了キューが運ぶものを、器にする（環-h・段い）。

QC の要素は `(語, 結果, 意図id, 種別, 番号)` の5つ組で、**位置で意味が決まっていた**。
種別は既に `完了` と `進捗` の2つあり、`進捗` では `結果`・`意図id`・`番号` が埋まらない。
**種別ごとに埋まる欄が違うものを、位置で運んでいた。**

環-h では主LLM の返りも QC へ載る（種別 `決定`）。運ぶのは `TurnResult`（道具の呼び出し）と、
**投げたときの W**（`memories`・`w_id_map`）である。6つ目・7つ目・8つ目を位置で足せば、
読む側が数えることになる。

段は（`Lookup`）と同じく、**1件を表す器**にする。**この段では挙動を変えない。**
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from familiar_agent.loop.event_loop import Trigger, Decision, InformationProcessing
from familiar_agent.loop.request import Request


def _ip():
    ip = InformationProcessing.__new__(InformationProcessing)
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._triggers = asyncio.Queue()
    ip._req.lookups = []
    ip._request_generation = 0
    ip._asyncio_loop = None
    return ip


# ── 器 ─────────────────────────────────────────────────────────────────────


def test_a_finished_lookup_carries_its_result():
    c = Trigger(kind="完了", query="明日の天気", result="晴れ", index=1)
    assert (c.kind, c.query, c.result, c.index) == ("完了", "明日の天気", "晴れ", 1)
    assert c.decision is None


def test_a_slow_notice_carries_only_the_query():
    """`進捗` は結果ではない。飛行中の数も一覧も触らない。"""
    c = Trigger(kind="進捗", query="明日の天気")
    assert c.result == ""
    assert c.index == 0


def test_a_decision_carries_the_turn_result_and_the_workspace():
    """主LLM の返りは、**投げたときの W** と一緒に運ぶ（環-h ②）。"""
    from familiar_agent.backends import ToolCall
    from familiar_agent.backends.types import TurnResult

    tr = TurnResult(
        stop_reason="tool_use", text="", tool_calls=[ToolCall("t", "say", {"text": "はい"})]
    )
    # **W は `Decision` が持つ。** `Trigger` の側に `memories` と `w_id_map` を並べると、
    # 種別が `決定` のときだけ意味を持つ欄が2つ増える。返りと W は必ず一緒に動くので、
    # 1つの器へまとめた（環-h ②）。
    d = Decision(
        result=tr,
        memories=[{"memory_id": "m1"}],
        w_id_map={"abcdef123456": "m1"},
        mem=MagicMock(),
        recent_ctx="",
        system=None,
        effort="high",
        capped=False,
    )
    c = Trigger(kind="決定", decision=d)
    assert c.decision is d
    assert c.decision.result is tr
    assert c.decision.memories == [{"memory_id": "m1"}]
    assert c.decision.w_id_map == {"abcdef123456": "m1"}


# ── 積む側が器を使うこと ────────────────────────────────────────────────────


def test_push_completion_puts_a_record():
    ip = _ip()
    ip.push_completion("明日の天気", "晴れ", index=3)
    got = ip._triggers.get_nowait()
    assert isinstance(got, Trigger)
    assert (got.kind, got.query, got.result, got.index) == ("完了", "明日の天気", "晴れ", 3)


def test_nothing_reads_the_queue_by_position():
    """位置で読む書き方が残っていないこと（`item[3]` のような）。"""
    import io
    import re
    import tokenize
    from pathlib import Path

    loop = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"
    with open(loop, "rb") as f:
        code = " ".join(
            t.string
            for t in tokenize.tokenize(io.BytesIO(f.read()).readline)
            if t.type not in (tokenize.COMMENT, tokenize.STRING)
        )
    # 5つ組をほどく書き方
    assert "for query , result_text , intent_id , _kind , _index in items" not in code
    assert not re.search(r"\bit \[ 3 \]", code)
