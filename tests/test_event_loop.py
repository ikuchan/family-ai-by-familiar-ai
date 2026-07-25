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
    a._expected_turns = len(list(stream_returns))
    a._turn_arousal = AsyncMock(return_value=0.3)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock(return_value=MagicMock())
    a.config = MagicMock()
    a.config.max_tokens = 400
    a.config.event_max_iterations = max_iters
    return a


def _run(a, utterance="こんにちは", on_text=None):
    return asyncio.run(InformationProcessing(a).run_iteration(utterance, on_text=on_text))


def _run_chain(a, utterance="こんにちは"):
    """人の発話で反復を起こし、駆動体が起こす続きの反復も終わるまで待って発話を返す。"""
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration(utterance, on_text=shown.append)
        for _ in range(400):
            if a.backend.stream_turn.await_count >= a._expected_turns and not ip._tasks:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)      # 最終反復の後始末が走るのを待つ
        await ip.close()

    asyncio.run(scenario())
    return "".join(shown)


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
    assert _run_chain(a) == "思い出したよ"
    assert a.backend.stream_turn.await_count == 2               # 2反復
    a._memory_tool.call.assert_awaited_once_with(                       # RH 実行
        "recall", {"query": "運動会"}, exclude_ids=["obs2"]
    )
    # QC drain＝完了結果を O へ書込（反復2の取込）。
    written = [c.args[0] for c in a._memory.save_async_with_id.call_args_list]
    assert any("recall結果テキスト" in w for w in written)   # 完了 O に結果が入る
    kinds = {c.kwargs["kind"] for c in a._memory.save_async_with_id.call_args_list}
    assert kinds == {"observation"}


def test_loop_records_form_a_single_chain():
    # ループ記録は1本の鎖：トリガO → 意図O → 完了O。新しい記録が直前の生きた記録を
    # supersede するので、生き残るのは常に鎖の先頭1件だけ。意図を書いた時点でトリガは
    # 死ぬので、その意図が出した検索にトリガは出てこない。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")
    calls = [c.args for c in a._memory.mark_superseded.call_args_list]
    assert calls == [("obs1", "obs2"), ("obs2", "obs3")]   # トリガ→意図→完了
    # ターン末に始末するのは鎖の先頭（＝生きている完了）だけ。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["superseded_ids"] == ["obs3"]


def test_w_search_excludes_the_intake_origin():
    # 一律の規則：取込で書いた記録（＝鎖の先頭）は検索から外す。素通しだと、問いと同一文の
    # トリガ O が必ず上位に来て、限られた枠から本物の記憶を押し出す。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    _, kwargs = a._active_memory().recall_async.call_args
    assert kwargs.get("exclude_ids") == ["obs1"]      # obs1＝取込で書いたトリガ O


def test_w_recall_query_follows_the_intake_origin():
    # 想起の手がかりは「取り込んだもの」＝鎖の先頭。反復2以降も最初の発話で探し続けると、
    # いま届いた完了とは無関係な検索になる（④ の「想起クエリ（手がかり）」）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")
    queries = [c.args[0] for c in a._active_memory().recall_async.call_args_list]
    assert queries[0] == "昨日の天気覚えてる？"                 # 反復1の起点＝人の発話
    assert "recall結果テキスト" in queries[1]                   # 反復2の起点＝完了O の内容


def test_w_recall_uses_configured_n():
    # 枠が 3 だと自己モデル文などが混じったとき本命が押し出される（実機で観測）。
    # 件数は Config で決める（既定 5）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a.config.recall_n = 5
    _run(a, utterance="おはよう")
    _, kwargs = a._active_memory().recall_async.call_args
    assert kwargs.get("n") == 5


def test_w_includes_the_intake_origin_deterministically():
    # 検索から外すかわりに、起点は W へ必ず加える（コサインの運に任せない）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    system = a.backend.stream_turn.call_args.kwargs["system"]
    assert "おはよう" in system                        # 起点（人の発話）
    assert "[想起]昔の話" in system                    # 想起結果も従来どおり


def test_recall_tool_excludes_the_intent_that_issued_it():
    # 意図 O は query を丸ごと含むので、その query で検索すれば必ず上位に来る（自己干渉）。
    # 自分が出した検索が自分自身を拾わないよう、意図 O の id だけ狭く除外する。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")
    _, kwargs = a._memory_tool.call.call_args
    assert kwargs["exclude_ids"] == ["obs2"]      # obs2＝この検索を出した意図 O


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
    _run_chain(a, utterance="おはよう")
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
    _run_chain(a, utterance="昨日の天気覚えてる？")
    # obs1=トリガ / obs2=open 意図 / obs3=完了 → 完了が意図を supersede（鎖の次）。
    assert ("obs2", "obs3") in [c.args for c in a._memory.mark_superseded.call_args_list]
    completions = [
        c for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "完了"
    ]
    assert len(completions) == 1
    content = completions[0].args[0]
    assert "昨日の天気" in content and "recall結果テキスト" in content


def test_intake_drains_inbox_in_place():
    # 駆動体は `self._inbox.append(await queue.get())` の append を **await の前に** 束縛する。
    # 取込が `_inbox` を作り直すと、駆動体は捨てられた古いリストへ積み、完了が失われる
    # （実機で観測：受領 inbox=0 → 取込 items=0）。同一オブジェクトを空にして守る。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    before = ip._inbox
    ip._inbox.append(("q", "結果", None))
    assert asyncio.run(ip._intake()) == 1
    assert ip._inbox is before        # 作り直さない
    assert ip._inbox == []            # 中身だけ空にする


def test_iteration_ends_when_tool_is_dispatched():
    # 1反復1出力：ツールを投げることも出力。投げた時点で反復は終わり、発話は持たない。
    a = _agent(stream_returns=[_turn([ToolCall(id="r", name="recall", input={"query": "q"})])])
    assert _run(a) == ""                       # 発話なしで反復終了
    a.backend.stream_turn.assert_awaited_once()  # 同じ呼び出しの中で次周回へ進まない


def test_driver_runs_next_iteration_when_completion_arrives():
    # 次の反復は完了が QC に届いて初めて起きる（駆動体・キュー到来で起きる）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        first = await ip.run_iteration("昨日の天気覚えてる？", on_text=shown.append)
        for _ in range(200):                    # 駆動体が起こす2反復目を待つ
            if shown:
                break
            await asyncio.sleep(0.01)
        return first

    assert asyncio.run(scenario()) == ""        # 1反復目は発話なし
    assert "".join(shown) == "晴れてたよ"       # 2反復目が発話した


def test_chain_cap_withholds_recall_tool():
    # 連鎖が上限に達した反復では recall を渡さない＝発話を必ず出す（暴走防止）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=2)
    _run_chain(a)
    assert a.backend.stream_turn.call_args_list[0].kwargs["tools"] == [_SAY_DEF, _RECALL_DEF]
    assert a.backend.stream_turn.call_args_list[1].kwargs["tools"] == [_SAY_DEF]


def test_recall_is_dispatched_async_and_loop_waits_on_queue():
    # RH（実行担当）が非同期に実行し、LPM は QC 到来で起きる。recall 本体が返らなくても、
    # 外から完了が届けばループは進む（同期 await のままでは進めない）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    never = asyncio.Event()

    async def hang(_name, _input):
        await never.wait()
        return ("届かない", None)

    a._memory_tool.call = AsyncMock(side_effect=hang)

    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        assert await ip.run_iteration("こんにちは", on_text=shown.append) == ""
        await asyncio.sleep(0.05)          # 意図を書いて dispatch し終えた頃
        ip._completion_queue.put_nowait(("q", "外から届いた結果", None))
        for _ in range(400):
            if shown:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert "".join(shown) == "はい"


def test_recall_iteration_does_not_display_filler_text():
    # 1反復1出力：say を決めた反復以外は表示しない（前置きの地の文が反復ごとに出て重複した）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})], text="まず記憶を探すね！"),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れ"})]),
    ])
    assert _run_chain(a) == "晴れ"        # 前置きは出さず、発話だけ1回
    # 生成中のストリームを止める＝呼び手の on_text を stream_turn へ渡さない。
    assert all(c.kwargs["on_text"] is None for c in a.backend.stream_turn.call_args_list)


def test_datetime_is_injected_into_prompt():
    # 日時が無いと「一昨日」「昨日」を自分で解けず、利用者に日付を聞き返す（実機で観測）。
    # 現行 run() と同じ書式 `(now :datetime "…")` で毎反復渡す。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a)
    system = a.backend.stream_turn.call_args.kwargs["system"]
    assert "(now :datetime " in system


def test_iteration_context_is_injected_into_prompt():
    # 反復番号と上限をコンテキストで渡す（あと何回で結論すべきかモデルが判断できる）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=3)
    _run_chain(a)
    systems = [c.kwargs["system"] for c in a.backend.stream_turn.call_args_list]
    assert "1/3" in systems[0]
    assert "2/3" in systems[1]


def test_all_loop_os_are_superseded_via_pipeline():
    # トリガ・open 意図・完了 のループ中 O は、すべてターン末に supersede される。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run_chain(a)
    _, kwargs = a._run_post_response_pipeline.call_args
    # 書いた O は3件（トリガ obs1・open 意図 obs2・完了 obs3）。意図 obs2 とトリガ obs1 は
    # 完了が解決済みなので、ターン末の一括 supersede には載せない（つながりを残すため）。
    assert a._memory.save_async_with_id.await_count == 3
    assert kwargs["superseded_ids"] == ["obs3"]


def test_max_iterations_bounds_the_chain():
    # 常に recall を返すモデルでも上限（2）で打ち切る（暴走防止）。
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        ],
        max_iters=2,
    )
    assert _run_chain(a) == ""                                 # 発話せず連鎖を閉じる
    assert a.backend.stream_turn.await_count == 2


# ── 診断ログ（反復・決定・上限空終了）─────────────────

_LOGGER = "familiar_agent.loop.event_loop"


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
        _run_chain(a)
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
        _run_chain(a)
    debugs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("iter=1/3" in m for m in debugs)
    assert any("iter=2/3" in m for m in debugs)   # 駆動体が起こした2反復目
