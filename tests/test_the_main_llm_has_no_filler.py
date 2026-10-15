"""主LLM の道具 `filler`（返事の前に言うつなぎ）を撤去した（出-cl 段 3・2026-10-15・本人の決定ア）。

考えているあいだは合図の声と待ちの電子音で分かるので、つなぎの一言はやめた。主LLM の `filler` は、手元のログ
（9/12〜10/15）で一度も使われていなかった。道具の定義、言ったつなぎを会話に置く口（各担い手の
`make_tool_call_message`）、つなぎの後に答えを待たせる間、重なった冒頭を削る処理も外す。O に残っている過去の
「つなぎ」の記録（役割 `つなぎ`）は読み続ける。
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from familiar_agent.backends import ToolCall
from familiar_agent.backends.types import TurnResult
from familiar_agent.config import AgentConfig
from familiar_agent.loop import event_loop
from familiar_agent.loop.event_loop import Decision, InformationProcessing
from familiar_agent.loop.request import Request
from familiar_agent.tools import tts
from tests.test_send_back_is_dispatched import _ip


def test_the_main_llm_is_offered_no_filler():
    assert "filler" not in event_loop._FULL_ACTIONS
    names = [
        d["name"] for d in tts.TTSTool.get_tool_definitions(SimpleNamespace(understands_tags=False))
    ]
    assert names == ["say"]


def _act(ip, result):
    return asyncio.run(
        ip._act_on_decision(
            Decision(
                result=result,
                memories=[],
                w_id_map={},
                verdict_view="__self__",
                recent_frame="",
                system=("安定", "可変"),
                effort="low",
                capped=False,
            ),
            utterance="こんにちは",
            gen=0,
        )
    )


def test_a_stray_filler_call_is_not_spoken():
    ip, _a = _ip()
    ip._req.trigger_kind = "情動"  # say の無い文の差し戻し（出-ck）を通らない起点で見る
    _act(
        ip,
        TurnResult(
            stop_reason="tool_use",
            text="",
            tool_calls=[ToolCall("f", "filler", {"text": "えっと"})],
        ),
    )
    ip._speak.assert_not_awaited()
    assert ip._finish.await_args.args[2] == "沈黙"


def test_a_say_is_still_spoken():
    """**反証の側**：返事（`say`）はいままでどおり声にする。"""
    ip, _a = _ip()
    _act(
        ip,
        TurnResult(
            stop_reason="tool_use",
            text="",
            tool_calls=[ToolCall("s", "say", {"text": "こんにちは"})],
        ),
    )
    ip._speak.assert_awaited_once()


def test_the_filler_machinery_is_gone():
    for name in (
        "_say_filler",
        "_filler_turns",
        "_pause_after_filler",
        "_speak_filler_in_background",
    ):
        assert not hasattr(InformationProcessing, name), name
    assert not hasattr(Request(), "said_fillers")
    assert not hasattr(tts, "FILLER_TOOL")
    from familiar_agent.io.dif import DIF

    assert not hasattr(DIF, "filler_defs")
    from familiar_agent.backends import anthropic, cli, gemini, glm, kimi, openai_compat

    for mod, cls in (
        (anthropic, "AnthropicBackend"),
        (cli, "CLIBackend"),
        (gemini, "GeminiBackend"),
        (glm, "GLMBackend"),
        (kimi, "KimiBackend"),
        (openai_compat, "OpenAICompatibleBackend"),
    ):
        assert not hasattr(getattr(mod, cls), "make_tool_call_message"), cls
    assert not hasattr(AgentConfig(), "filler_answer_gap_seconds")
    with pytest.raises(ImportError):
        importlib.import_module("familiar_agent.core.filler_echo")
