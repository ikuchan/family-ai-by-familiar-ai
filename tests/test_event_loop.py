"""#11 段階1：run_iteration が「想起→1回生成（ツール無し）→1発話」を返し永続化を回す。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backend import ToolCall, TurnResult
from familiar_agent.loop.event_loop import run_iteration

_SAY_DEF = {"name": "say", "input_schema": {}}


def _agent(*, tool_calls, text=""):
    a = MagicMock()
    a._me_md = "[ME] 口調"
    a._family_md = "[FAMILY] 家族"
    mem = MagicMock()
    mem.recall_async = AsyncMock(return_value=[{"memory_id": "m1", "summary": "昔の話"}])
    mem.format_for_context = MagicMock(return_value="[想起]昔の話")
    a._active_memory = MagicMock(return_value=mem)
    a._pmm = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=[])
    a._tts = MagicMock()
    a._tts.get_tool_definitions = MagicMock(return_value=[_SAY_DEF])
    a._tts.call = AsyncMock(return_value=("ok", None))
    a.backend = MagicMock()
    a.backend.make_user_message = MagicMock(return_value={"role": "user", "content": "こんにちは"})
    a.backend.stream_turn = AsyncMock(
        return_value=(TurnResult(stop_reason="end_turn", text=text, tool_calls=tool_calls), {})
    )
    a._turn_arousal = AsyncMock(return_value=0.3)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock(return_value=MagicMock())
    a.config = MagicMock()
    a.config.max_tokens = 400
    return a


def test_run_iteration_speaks_via_say_tool():
    a = _agent(tool_calls=[ToolCall(id="t", name="say", input={"text": "やあ、元気？"})])
    out = asyncio.run(run_iteration(a, "こんにちは"))
    assert out == "やあ、元気？"                              # say の text を発話として返す
    a._active_memory().recall_async.assert_awaited_once()   # 想起1回（拡散込み）
    a.backend.stream_turn.assert_awaited_once()              # 生成1回（多段しない）
    a._tts.call.assert_awaited_once_with("say", {"text": "やあ、元気？"})  # TTS 実行
    # 段階1は say ツールだけを渡す（発話のみ・多段 ReAct を構造的に禁止）。
    _, kwargs = a.backend.stream_turn.call_args
    assert kwargs.get("tools") == [_SAY_DEF]
    assert "max_tokens" in kwargs and kwargs["max_tokens"] == 400  # 必須引数を渡す（回帰防止）
    assert "on_text" in kwargs                                     # ストリーミング先を渡す
    a._spawn_background_task.assert_called_once()            # 永続化を回す


def test_run_iteration_takes_first_say_and_suppresses_duplicate():
    a = _agent(
        tool_calls=[
            ToolCall(id="t", name="say", input={"text": "先頭だけ"}),
            ToolCall(id="t", name="say", input={"text": "重複は捨てる"}),
        ]
    )
    out = asyncio.run(run_iteration(a, "こんにちは"))
    assert out == "先頭だけ"                                  # run() と同じ「先頭 say 採用」
    a._tts.call.assert_awaited_once_with("say", {"text": "先頭だけ"})  # 重複 say は実行しない


def test_run_iteration_falls_back_to_text_when_no_say():
    a = _agent(tool_calls=[], text="ツール無しの素テキスト")
    out = asyncio.run(run_iteration(a, "こんにちは"))
    assert out == "ツール無しの素テキスト"                    # say が無ければ result.text へ保険
    a._tts.call.assert_not_awaited()
