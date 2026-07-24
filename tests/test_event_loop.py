"""#11 段階1：I（情報処理機構）の LPM 反復。

スライス1＝人の発言→想起→1発話（say 経由）。
スライス2＝内部ツール recall を QC（完了キュー）経由で O→W 連鎖し、消化した完了 O は
ターン観察で supersede、上限は Config で打ち切る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backend import ToolCall, TurnResult
from familiar_agent.loop.event_loop import InformationProcessing

_SAY_DEF = {"name": "say", "input_schema": {}}
_RECALL_DEF = {"name": "recall", "input_schema": {}}
_REMEMBER_DEF = {"name": "remember", "input_schema": {}}


def _turn(tool_calls, text=""):
    return (TurnResult(stop_reason="end_turn", text=text, tool_calls=tool_calls), {})


def _agent(*, stream_returns, max_iters=3):
    """stream_returns：stream_turn の各反復の戻り（_turn(...) のリスト）。"""
    a = MagicMock()
    a._me_md = "[ME] 口調"
    a._family_md = "[FAMILY] 家族"
    mem = MagicMock()
    mem.recall_async = AsyncMock(return_value=[{"memory_id": "m1", "summary": "昔の話"}])
    mem.format_for_context = MagicMock(return_value="[想起]昔の話")
    a._active_memory = MagicMock(return_value=mem)
    a._memory = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs123", True))
    a._memory.mark_superseded = MagicMock()
    a._observation_perspective = MagicMock(return_value={})
    a._memory_tool = MagicMock()
    a._memory_tool.get_tool_definitions = MagicMock(
        return_value=[_REMEMBER_DEF, _RECALL_DEF, {"name": "note_to_share"}]
    )
    a._memory_tool.call = AsyncMock(return_value=("recall結果テキスト", None))
    a._pmm = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=[])
    a._tts = MagicMock()
    a._tts.get_tool_definitions = MagicMock(return_value=[_SAY_DEF])
    a._tts.call = AsyncMock(return_value=("ok", None))
    a.backend = MagicMock()
    a.backend.make_user_message = MagicMock(return_value={"role": "user", "content": "x"})
    a.backend.stream_turn = AsyncMock(side_effect=list(stream_returns))
    a._turn_arousal = AsyncMock(return_value=0.3)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock(return_value=MagicMock())
    a.config = MagicMock()
    a.config.max_tokens = 400
    a.config.event_max_iterations = max_iters
    return a


def _run(a, utterance="こんにちは", on_text=None):
    return asyncio.run(InformationProcessing(a).run_iteration(utterance, on_text=on_text))


# ── スライス1（発話のみ）─────────────────────────────

def test_speaks_via_say_tool():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ、元気？"})])])
    out = _run(a)
    assert out == "やあ、元気？"
    a._active_memory().recall_async.assert_awaited_once()
    a.backend.stream_turn.assert_awaited_once()
    a._tts.call.assert_awaited_once_with("say", {"text": "やあ、元気？"})
    _, kwargs = a.backend.stream_turn.call_args
    assert kwargs.get("tools") == [_SAY_DEF, _RECALL_DEF]   # say＋recall のみ
    assert kwargs["max_tokens"] == 400
    assert "on_text" in kwargs
    a._memory.save_async_with_id.assert_not_awaited()       # 完了 O 書込みは無い
    a._spawn_background_task.assert_called_once()


def test_takes_first_say_and_suppresses_duplicate():
    a = _agent(stream_returns=[_turn([
        ToolCall(id="t", name="say", input={"text": "先頭だけ"}),
        ToolCall(id="t", name="say", input={"text": "重複は捨てる"}),
    ])])
    assert _run(a) == "先頭だけ"
    a._tts.call.assert_awaited_once_with("say", {"text": "先頭だけ"})


def test_falls_back_to_text_when_no_tool():
    a = _agent(stream_returns=[_turn([], text="ツール無しの素テキスト")])
    assert _run(a) == "ツール無しの素テキスト"
    a._tts.call.assert_not_awaited()


def test_emits_say_text_to_on_text_for_display():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    shown: list[str] = []
    _run(a, on_text=shown.append)
    assert "やあ" in "".join(shown)


def test_does_not_double_emit_fallback_text():
    a = _agent(stream_returns=[_turn([], text="素テキスト")])
    shown: list[str] = []
    _run(a, on_text=shown.append)
    assert "".join(shown) == ""


# ── スライス2（QC 連鎖・supersede・上限）─────────────

def test_recall_chains_via_completion_queue_then_says():
    # 反復1＝recall を呼ぶ／反復2＝say。RH が recall を実行→QC→次反復で O 書込→発話。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "運動会"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "思い出したよ"})]),
    ])
    out = _run(a)
    assert out == "思い出したよ"
    assert a.backend.stream_turn.await_count == 2               # 2反復
    a._memory_tool.call.assert_awaited_once_with("recall", {"query": "運動会"})  # RH 実行
    # QC drain＝完了結果を O へ書込（反復2の取込）。
    a._memory.save_async_with_id.assert_awaited_once()
    args, kwargs = a._memory.save_async_with_id.call_args
    assert args[0] == "recall結果テキスト"
    assert kwargs["kind"] == "observation"


def test_consumed_completion_is_superseded_via_pipeline():
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run(a)
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["superseded_ids"] == ["obs123"]              # 消化した完了 O id


def test_max_iterations_bounds_the_chain():
    # 常に recall を返すモデルでも上限（2）で打ち切る（暴走防止）。
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        ],
        max_iters=2,
    )
    out = _run(a)
    assert out == ""                                           # say せず打ち切り
    assert a.backend.stream_turn.await_count == 2
