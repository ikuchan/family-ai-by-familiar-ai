"""#11 段階1：run_iteration が「想起→1回生成（ツール無し）→1発話」を返し永続化を回す。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backend import TurnResult
from familiar_agent.loop.event_loop import run_iteration


def _agent():
    a = MagicMock()
    a._me_md = "[ME] 口調"
    a._family_md = "[FAMILY] 家族"
    mem = MagicMock()
    mem.recall_async = AsyncMock(return_value=[{"memory_id": "m1", "summary": "昔の話"}])
    mem.format_for_context = MagicMock(return_value="[想起]昔の話")
    a._active_memory = MagicMock(return_value=mem)
    a._pmm = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=[])
    a.backend = MagicMock()
    a.backend.make_user_message = MagicMock(return_value={"role": "user", "content": "こんにちは"})
    a.backend.stream_turn = AsyncMock(
        return_value=(TurnResult(stop_reason="end_turn", text="やあ、元気？", tool_calls=[]), {})
    )
    a._turn_arousal = AsyncMock(return_value=0.3)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock(return_value=MagicMock())
    a.config = MagicMock()
    a.config.max_tokens = 400
    return a


def test_run_iteration_recalls_generates_once_and_returns_speech():
    a = _agent()
    out = asyncio.run(run_iteration(a, "こんにちは"))
    assert out == "やあ、元気？"
    a._active_memory().recall_async.assert_awaited_once()   # 想起1回（拡散込み）
    a.backend.stream_turn.assert_awaited_once()              # 生成1回（多段しない）
    # 段階1は発話のみ＝ツールを渡さない（1出力保証）。
    _, kwargs = a.backend.stream_turn.call_args
    assert kwargs.get("tools") == []
    assert "max_tokens" in kwargs and kwargs["max_tokens"] == 400  # 必須引数を渡す（回帰防止）
    assert "on_text" in kwargs                                     # ストリーミング先を渡す
    a._spawn_background_task.assert_called_once()            # 永続化を回す
