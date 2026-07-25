"""#11 段階1：I（情報処理機構）の LPM 反復。

スライス1＝人の発言→想起→1発話（say 経由）。
スライス2＝内部ツール recall を QC（完了キュー）経由で O→W 連鎖し、消化した完了 O は
ターン観察で supersede、上限は Config で打ち切る。
"""

from __future__ import annotations

import asyncio
import logging
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
    # 書込みごとに別 id を返す（トリガ／open 意図／完了 を区別して検証するため）。
    _ids = iter([f"obs{i}" for i in range(1, 20)])
    a._memory.save_async_with_id = AsyncMock(side_effect=lambda *_a, **_k: (next(_ids), True))
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
    # 取込でトリガ（発話）O を1件書くだけ（open 意図・完了 O は無い）。
    assert a._memory.save_async_with_id.await_count == 1
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


def test_fallback_text_emitted_once():
    # 生成中はストリームせず、決定後に1回だけ出す（1反復1出力）。
    a = _agent(stream_returns=[_turn([], text="素テキスト")])
    shown: list[str] = []
    _run(a, on_text=shown.append)
    assert "".join(shown) == "素テキスト"


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
    written = [c.args[0] for c in a._memory.save_async_with_id.call_args_list]
    assert any("recall結果テキスト" in w for w in written)   # 完了 O に結果が入る
    kinds = {c.kwargs["kind"] for c in a._memory.save_async_with_id.call_args_list}
    assert kinds == {"observation"}


def test_trigger_utterance_written_to_o_at_intake():
    # 取込＝来た事実（人の発話）を O に書く（④シーケンス）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    _run(a, utterance="おはよう")
    first = a._memory.save_async_with_id.call_args_list[0]
    assert first.args[0] == "おはよう"
    assert first.kwargs["direction"] == "発話"


def test_open_intent_written_with_utterance_and_query():
    # recall を決めた反復で open 意図 O を書く。content は元の発話内容と query を含む
    # （id ではなく内容そのもの＝W に載ったとき意味が通り想起にも効く）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "運動会"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run(a, utterance="おはよう")
    intents = [
        c for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "意図"
    ]
    assert len(intents) == 1
    content = intents[0].args[0]
    assert "おはよう" in content and "運動会" in content


def test_completion_supersedes_open_intent_and_records_search():
    # 完了は open 意図に「再会」して解決する（[D-単一想起]）。完了 O が意図 O を supersede し、
    # content は「探した事実＋結果」を持つ。これが無いと W に「結果はまだ無い」が残り続け、
    # モデルは同じ recall を繰り返す（実機で観測）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run(a, utterance="昨日の天気覚えてる？")
    # obs1=トリガ / obs2=open 意図 / obs3=完了 → 完了が意図を supersede。
    a._memory.mark_superseded.assert_called_once_with("obs2", "obs3")
    completions = [
        c for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "完了"
    ]
    assert len(completions) == 1
    content = completions[0].args[0]
    assert "昨日の天気" in content and "recall結果テキスト" in content


def test_new_intent_supersedes_still_live_previous_intent():
    # 意図は常に高々1件。書込み時点で、まだ生きている前の意図を supersede する。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    ip = InformationProcessing(a)
    ip._live_intent_id = "old-intent"
    asyncio.run(ip.run_iteration("こんにちは"))
    # obs1=トリガ / obs2=新しい意図。
    assert ("old-intent", "obs2") in [c.args for c in a._memory.mark_superseded.call_args_list]


def test_resolved_intent_is_not_superseded_again():
    # 完了が解決した意図を、次の意図書込みで上書きしない（解決のつながりを残す）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q1"})]),
        _turn([ToolCall(id="r", name="recall", input={"query": "q2"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=3)
    _run(a)
    calls = [c.args for c in a._memory.mark_superseded.call_args_list]
    assert ("obs2", "obs3") in calls                              # 完了 obs3 が意図 obs2 を解決
    assert not any(c[0] == "obs2" and c[1] != "obs3" for c in calls)


def test_intent_records_cap_reached_on_last_iteration():
    # 上限に達した反復で書く意図は「これ以上探さない」と持つ。次ターンの W で
    # 「結果はまだ無い」と読まれて再検索が繰り返される自己増殖を止める。
    a = _agent(
        stream_returns=[_turn([ToolCall(id="r", name="recall", input={"query": "q"})])],
        max_iters=1,
    )
    _run(a)
    intents = [
        c for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "意図"
    ]
    assert "上限" in intents[0].args[0]
    assert "結果はまだ無い" not in intents[0].args[0]


def test_recall_iteration_does_not_display_filler_text():
    # 1反復1出力：say を決めた反復以外は表示しない（前置きの地の文が反復ごとに出て重複した）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})], text="まず記憶を探すね！"),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れ"})]),
    ])
    shown: list[str] = []
    _run(a, on_text=shown.append)
    assert "".join(shown) == "晴れ"       # 前置きは出さず、発話だけ1回
    # 生成中のストリームを止める＝呼び手の on_text を stream_turn へ渡さない。
    assert all(c.kwargs["on_text"] is None for c in a.backend.stream_turn.call_args_list)


def test_iteration_context_is_injected_into_prompt():
    # 反復番号と上限をコンテキストで渡す（あと何回で結論すべきかモデルが判断できる）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=3)
    _run(a)
    systems = [c.kwargs["system"] for c in a.backend.stream_turn.call_args_list]
    assert "1/3" in systems[0]
    assert "2/3" in systems[1]


def test_all_loop_os_are_superseded_via_pipeline():
    # トリガ・open 意図・完了 のループ中 O は、すべてターン末に supersede される。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run(a)
    _, kwargs = a._run_post_response_pipeline.call_args
    # 書いた O は3件（トリガ obs1・open 意図 obs2・完了 obs3）。意図 obs2 は完了が解決済みなので
    # ターン末の一括 supersede には載せない（「完了が意図を解決した」つながりを残すため）。
    assert a._memory.save_async_with_id.await_count == 3
    assert kwargs["superseded_ids"] == ["obs1", "obs3"]


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


# ── 診断ログ（反復・決定・上限空終了）─────────────────

_LOGGER = "familiar_agent.loop.event_loop"


def test_warns_when_cap_reached_without_say(caplog):
    # 上限まで recall を返し say 未決で終わる＝空応答経路 → WARNING が出る。
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        ],
        max_iters=2,
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        _run(a)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("上限" in r.getMessage() for r in warns)


def test_no_warning_on_normal_say(caplog):
    # 通常の発話（say）では上限空終了の WARNING は出ない。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        _run(a)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("上限" in r.getMessage() for r in warns)


def test_info_summary_reports_iteration_count(caplog):
    # ターン終了時の INFO 総括に反復数が載る（本番ログで再構成できる）。
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
        ]
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _run(a)
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("反復=2" in m for m in infos)


def test_debug_lines_carry_iteration_number(caplog):
    # debug 行に反復番号（iter=N/M）が付き、どの反復か分かる。
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
        ]
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        _run(a)
    debugs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("iter=1/3" in m for m in debugs)
    assert any("iter=2/3" in m for m in debugs)
