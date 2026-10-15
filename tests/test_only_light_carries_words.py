"""調停の決定で言葉を持つのは light の返事（`text`）だけ。つなぎ（`filler`）と捨て場（`trash`）は書かせない。

つなぎは出-aj（2026-09-23）で返事とは別の欄に分け、出-aq 段 7 で調停の決定からは書かせなくし、出-cl 段 2
（2026-10-15・本人の決定ア）で待ちの知らせのつなぎもやめた。full と action の決定は言葉を持たない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput, assemble


def _d(data: dict):
    """Jev の答えと軽量LLM の文章を合わせた辞書を、決定に組み立てる。"""
    return assemble(data, can_see=False, origin="発話")


def test_a_light_turn_speaks_the_text():
    got = _d({"branch": "light", "text": "おかえりなさい。"})
    assert got is not None and got.text == "おかえりなさい。"


def test_a_light_turn_ignores_a_filler():
    got = _d({"branch": "light", "text": "おかえりなさい。", "filler": "まちがい"})
    assert got is not None and got.text == "おかえりなさい。"


def test_full_and_action_turns_carry_no_words():
    """**反証の側**：filler や text が混じっていても、full と action は言葉を持たない。"""
    full = _d({"branch": "full", "effort": "medium", "filler": "こんにちは。", "text": "用件"})
    act = _d({"branch": "action", "action": "recall", "query": "本", "filler": "調べますね。"})
    assert full is not None and full.text == ""
    assert act is not None and act.text == ""


def _asked_for(data: dict) -> str:
    w = MagicMock()
    w.complete = AsyncMock(return_value='{"text": "うん"}')
    inp = ArbiterInput(utterance="こんにちは", workspace_ctx="（なし）")
    asyncio.run(Arbiter(jev=None, writer=w, timeout=2.0)._write(inp, data))
    return w.complete.await_args.args[0]


def test_the_writer_is_never_asked_for_a_filler():
    for data in ({"branch": "action", "action": "search_deferred"}, {"branch": "light"}):
        asked = _asked_for(data)
        assert '"filler"' not in asked and '"trash"' not in asked
