"""記憶の申告（軽量LLM が答えて閉じた反復）を Jev が決める（出-au 段 5-5・2026-09-27・`設計方針_判定の段` §2.2.3）。

主LLM の申告（`say` の `memory_verdicts`）は主LLM 自身がするので移さない。移すのは、調停の light で軽量LLM が答えて
閉じた反復の申告（出-h-ろ）だけ。

- W の過去の記憶（`w_id_map` の id）ごとに 1 問、**1 回の呼び出しでまとめて**聞く。選択肢は important／referred／
  useless／unused（主LLM と同じ 4 通り・後ろの根づきの更新がこれを使う）。
- 返りは `[{"id": 12桁, "verdict": …}]`（主LLM の申告と同じ形）。確信度 0.6 未満の記憶は申告しない。
- Jev が使えない・失敗なら空（申告しない＝倒し先）。軽量LLM の口（`workspace.ask_verdicts`）は外した。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges

_W = """[過去の記憶]
- 2025-08-03 12:00 id:abcdef123456 (適合度:0.8) 海で泳げるようになった
- 2025-08-04 12:00 id:111122223333 (適合度:0.4) 山に行った"""
_IDS = {"abcdef123456": "abcdef123456-x", "111122223333": "111122223333-x"}


def _client(answers=None, *, ok=True):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers or {}))
    return c


def _judge(client):
    return asyncio.run(
        jev_judges.judge_verdicts(
            client,
            utterance="海って楽しいよね",
            reply="楽しいですよね。",
            workspace_ctx=_W,
            ids=list(_IDS),
            min_conf=0.6,
        )
    )


def test_each_memory_gets_one_question_in_one_call():
    c = _client(
        {
            "abcdef123456": {"choice": "referred", "confidence": 0.9},
            "111122223333": {"choice": "unused", "confidence": 0.8},
        }
    )
    got = _judge(c)
    assert got == [
        {"id": "abcdef123456", "verdict": "referred"},
        {"id": "111122223333", "verdict": "unused"},
    ]
    c.ask.assert_awaited_once()
    state, questions = c.ask.await_args.args
    assert set(questions) == set(_IDS)
    assert set(questions["abcdef123456"]["criteria"]) == {
        "important",
        "referred",
        "useless",
        "unused",
    }
    assert "海で泳げるようになった" in questions["abcdef123456"]["instructions"]
    assert "海って楽しいよね" in state and "楽しいですよね。" in state


def test_low_confidence_memories_are_not_declared():
    got = _judge(_client({"abcdef123456": {"choice": "important", "confidence": 0.5}}))
    assert got == []


def test_failure_declares_nothing():
    assert _judge(_client(ok=False)) == []
    assert _judge(None) == []


def test_the_light_close_asks_jev_and_the_light_llm_judge_is_gone():
    from familiar_agent.loop import workspace
    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._declare_light_memory_use)
    assert "jev_judges.judge_verdicts" in src
    assert not hasattr(workspace, "ask_verdicts")
    assert not hasattr(workspace, "_VERDICT_PROMPT")
