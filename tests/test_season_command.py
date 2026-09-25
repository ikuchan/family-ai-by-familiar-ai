"""`/season clear`——いまの季節とまわりを消す（知-ac 段 5・2026-09-26・`設計方針_季節の層` v0.1 §2）。

人が手で直す道は**消すことだけ**（本人の決定）。間違った中身を渡さないようにし、次に季節の層が
回った晩に書き直される。LLM を通さない（`/timer stop`・`/mic on` と同じ口）。消したあとは暦だけが渡る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.core import season_env as se


def _agent():
    a = MagicMock(spec=Agent)
    return a


def test_season_clear_clears(monkeypatch):
    cleared: list[int] = []
    monkeypatch.setattr(se, "clear", lambda: cleared.append(1) or True)
    reply = Agent._handle_season_command(_agent(), "/season clear")
    assert cleared == [1]
    assert "消した" in reply


def test_a_tail_is_tolerated(monkeypatch):
    """音声やかな入力の句読点（`/timer resume・`）と同じく、末尾の記号は許す。"""
    monkeypatch.setattr(se, "clear", lambda: True)
    assert Agent._handle_season_command(_agent(), "/season clear。") is not None


def test_other_text_is_not_a_season_command(monkeypatch):
    monkeypatch.setattr(se, "clear", lambda: (_ for _ in ()).throw(AssertionError("消した")))
    assert Agent._handle_season_command(_agent(), "季節の話をしよう") is None
    assert Agent._handle_season_command(_agent(), "/season") is None


def test_a_failed_clear_says_so(monkeypatch):
    monkeypatch.setattr(se, "clear", lambda: False)
    assert "消せなかった" in Agent._handle_season_command(_agent(), "/season clear")


def test_run_answers_without_the_llm(monkeypatch):
    """`run()` は命令を LLM へ渡さずに返す。"""
    from tests.test_input_commands_before_loop_branch import _agent as loop_agent

    monkeypatch.setattr(se, "clear", lambda: True)
    a = loop_agent()
    a._handle_season_command = lambda ui: Agent._handle_season_command(a, ui)
    reply = asyncio.run(Agent.run(a, "/season clear"))
    assert "消した" in reply
    a._info_processing.push_utterance.assert_not_awaited()


def test_the_loop_stub_knows_the_command():
    """ほかの命令の試験のスタブは、この口を None で素通りさせる。"""
    from tests.test_input_commands_before_loop_branch import _agent as loop_agent

    assert loop_agent()._handle_season_command("/speaker パパ") is None
