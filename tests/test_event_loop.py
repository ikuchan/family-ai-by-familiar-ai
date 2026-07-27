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
_SEARCH_DEF = {"name": "search_deferred", "input_schema": {}}
_FETCH_DEF = {"name": "fetch_deferred", "input_schema": {}}


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
    a._memory.close_with_children = MagicMock()
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
    a._utility_backend = MagicMock()
    a._utility_backend.complete = AsyncMock(return_value='{"branch":"full","effort":"high"}')
    a._expected_turns = len(list(stream_returns))
    a._social_presence_permission = MagicMock(return_value=1.0)   # 既定＝誰か居る
    a._in_quiet_hours = MagicMock(return_value=False)             # 既定＝静穏時間ではない
    a._deferred_search = MagicMock()
    a._deferred_search.get_tool_definitions = MagicMock(return_value=[_SEARCH_DEF])
    a._deferred_search.call = AsyncMock(return_value=("投げた", None))
    a._deferred_fetch = MagicMock()
    a._deferred_fetch.get_tool_definitions = MagicMock(return_value=[_FETCH_DEF])
    a._deferred_fetch.call = AsyncMock(return_value=("投げた", None))
    a._pending_store = MagicMock()
    a._pending_store.add = MagicMock(return_value="pending-1")
    a._turn_arousal = AsyncMock(return_value=0.3)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock(return_value=MagicMock())
    a.config = MagicMock()
    a.config.max_tokens = 400
    a.config.event_max_iterations = max_iters
    # 数値として使う設定は明示する。MagicMock のままだと `content[:cap]` の cap が
    # `__index__`=1 と解釈され、content が黙って1文字に切られる（実際に起きた）。
    a.config.completion_content_max = 8192
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
    # 発話・記憶・net（投げっぱなしの外部呼び出し）を渡す。
    assert kwargs.get("tools") == [_SAY_DEF, _RECALL_DEF, _SEARCH_DEF, _FETCH_DEF]
    assert kwargs["max_tokens"] == 400
    assert "on_text" in kwargs
    # 取込でトリガ（発話）O、発話時点で本応答 O の2件（open 意図・完了 O は無い）。
    # 本応答を背景の永続化に任せると2秒遅れ、次の反復が「さっき何と言ったか」を拾えない。
    assert a._memory.save_async_with_id.await_count == 2
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
    # ターン末に生き残るのは本応答 O（obs4）だけ。求めを閉じる側がこの記録なので、
    # 意図・完了はここで閉じ、この1件が次の反復の候補に残る。要約が後から supersede する。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["superseded_ids"] == ["obs4"]


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


def test_children_are_linked_to_the_parent_and_closed_together():
    # 親＝人の求め、子＝その求めのために投げた調査。子は parent_id で親に紐づき、
    # 答えて親が決着したら、生きている子もまとめて閉じる（一段だけ・再帰なし）。
    # deferred はモックだと完了が届かない（実体が sink を呼ぶ）ので、同期で返る recall で試す。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "今日の天気"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})]),
    ])
    assert _run_chain(a, utterance="今日の天気を調べて") == "晴れだよ"
    intents = [c for c in a._memory.save_async_with_id.call_args_list
               if c.kwargs.get("direction") == "意図"]
    assert intents and intents[0].kwargs.get("parent_id") == "obs1"   # obs1＝親（求め）
    # 実際に閉じるのは永続化パイプライン（ここではモック）。親の id が渡ることを見る。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["close_parent_id"] == "obs1"


def test_completion_content_reads_as_this_chains_action():
    # 完了 MI は「何を・どうやって調べた結果が届いたか」が content から読めること。
    # 「探した結果：…」だけだと、自分がいましたことなのか昔の記憶なのか区別できず、
    # 結果が W にあるのに調停がまた調べに行った（実機で観測）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="search_deferred", input={"query": "今日の天気"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
    ])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("今日の天気を調べて")
        ip.push_completion("今日の天気", "西日本は暑い")
        for _ in range(200):
            if any(c.kwargs.get("direction") == "完了"
                   for c in a._memory.save_async_with_id.call_args_list):
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    done = next(c for c in a._memory.save_async_with_id.call_args_list
                if c.kwargs.get("direction") == "完了")
    content = done.args[0]
    assert "search_deferred" in content      # どうやって調べたか
    assert "今日の天気" in content            # 何を
    assert "届いた" in content                # いま届いたこと


def test_completion_content_keeps_the_fetched_body_up_to_the_embedding_limit():
    # 取ってきた本文を 500 字で切ると、表なら見出しだけが残って中身が消える（実機で観測：
    # フルLLM が「データが読み取れなかった」と正しく報告した）。上限は埋め込みモデル
    # bge-m3 の入力上限 8192 トークンに合わせる。1文字＝1トークンになる字もあるので、
    # 8192 *文字* なら常に 8192 トークン以下に収まり、埋め込みが後ろを落とさない。
    body = "気" * 6000                       # 500 でも 8192 でもない長さ
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="search_deferred", input={"query": "今日の天気"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
    ])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("今日の天気を調べて")
        ip.push_completion("今日の天気", body)
        for _ in range(200):
            if any(c.kwargs.get("direction") == "完了"
                   for c in a._memory.save_async_with_id.call_args_list):
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    done = next(c for c in a._memory.save_async_with_id.call_args_list
                if c.kwargs.get("direction") == "完了")
    assert done.args[0].count("気") >= 6000   # 本文が丸ごと残る
    assert len(done.args[0]) <= 8192          # 埋め込みの入力上限は超えない


def test_capped_iteration_tells_the_full_llm_to_admit_it_could_not_finish():
    # 上限では、黙って手持ちで繕わず「調べきれなかった」と断ってから分かることを返す。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])],
               max_iters=1)
    _run(a, utterance="調べて")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "上限に達した" in system and "現時点で分かること" in system


def test_workspace_records_are_notes_not_a_script_to_read_aloud():
    # W に載るループの記録は自分の覚え書き。内部の言い回し（「調べた結果が届いた」）を
    # そのまま復唱した（実機で観測）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "読み上げる文ではない" in system


def test_w_presents_the_loop_records_as_mi_not_synthetic_labels():
    # W は「思い出している記憶」ではなく、いまの作業状態。ループの記録は MI としてそのまま
    # 並べ、合成ラベル（[取込]・[調査中]）は作らない。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[取込]" not in system and "[調査中]" not in system
    assert "おはよう" in system               # MI の内容自体は載る


def test_w_includes_the_intake_origin_deterministically():
    # 検索から外すかわりに、起点は W へ必ず加える（コサインの運に任せない）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
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


def test_system_prompt_is_split_for_caching():
    # 安定部（静的核＋ME＋FAMILY＋capabilities）と可変部を分けて渡すと、安定部に
    # cache_control が付いて反復ごとの再処理が減る。1本の文字列だと効かない。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a)
    system = a.backend.stream_turn.call_args.kwargs["system"]
    assert isinstance(system, tuple) and len(system) == 2
    assert "[ME] 口調" in system[0]          # 安定部
    assert "[想起]昔の話" in system[1]        # 可変部


def test_light_branch_speaks_without_the_full_llm():
    # 軽量LLM が「短文で足りる」と判断した反復は、フルLLM を呼ばずに閉じる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "使わない"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"light","text":"やあ！元気？"}')
    assert _run(a) == "やあ！元気？"
    a.backend.stream_turn.assert_not_awaited()      # フルLLM を起こさない
    a._tts.call.assert_awaited_once_with("say", {"text": "やあ！元気？"})


def test_action_branch_dispatches_recall_without_the_full_llm():
    # 探すと決まっている反復も、フルLLM を起こさずに recall を投げて閉じる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "使わない"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","query":"昨日の天気"}')
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        first = await ip.run_iteration("こんにちは", on_text=shown.append)
        # 反復1でフルLLM を起こしていないこと（この後、駆動体が続きの反復を回す）。
        assert a.backend.stream_turn.await_count == 0
        for _ in range(200):
            if a._memory_tool.call.await_count:
                break
            await asyncio.sleep(0.005)
        query = a._memory_tool.call.await_args.args[1]
        await ip.close()
        return first, query

    first, query = asyncio.run(scenario())
    assert first == ""                                   # 発話を持たない反復
    assert query == {"query": "昨日の天気"}              # 調停が決めた語で投げる


def test_full_branch_passes_the_effort_chosen_by_the_arbiter():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"full","effort":"low"}')
    _run(a)
    assert a.backend.stream_turn.call_args.kwargs["effort"] == "low"


def test_human_utterance_marks_presence_before_the_gate():
    # 人が話しかけた時点で在席は立つ。反復に入る前に印を付けないと、応答前の在席判定が
    # 「誰も居ない」になり、目の前の相手への返事まで保留になる（実機で観測）。
    # 印を発話の受領時に付けることで、連鎖が長引いて相手が去った場合は在席が切れる
    # （その場合は独り言にならず保留になる）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    a._last_human_at = 0.0
    _run(a, utterance="こんばんは")
    assert a._last_human_at > 0.0          # 受領時に更新される


def test_speech_is_held_as_pending_when_nobody_is_present():
    # 身体を持つ以上、発話は聞く相手が居て初めて意味を持つ。居なければ話さず、
    # 「話したかったができなかった」を pending_speech に積んで反復を終える。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "ねえ聞いて"})])])
    a._social_presence_permission = MagicMock(return_value=0.0)   # 誰も居ない
    shown: list[str] = []
    assert _run(a, on_text=shown.append) == ""       # 発話しない
    a._tts.call.assert_not_awaited()                  # 音も出さない
    assert shown == []                                # 画面にも出さない
    a._pending_store.add.assert_called_once()         # 後で話すために積む


def test_speech_goes_out_when_someone_is_present():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "ねえ聞いて"})])])
    assert _run(a) == "ねえ聞いて"
    a._tts.call.assert_awaited_once_with("say", {"text": "ねえ聞いて"})
    a._pending_store.add.assert_not_called()          # 話せたので溜めない


def test_deferred_is_wired_to_the_completion_queue_only_when_on():
    # EVENT_LOOP on のときだけ deferred の完了をキューへ渡す。off では従来どおり溜めて
    # ポーリングで拾われる（二重配信にしない＝排他）。
    from familiar_agent.agent import EmbodiedAgent

    async def build(event_loop_on: bool):
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.event_loop = event_loop_on
        agent._info_processing = None
        agent._tonic = None
        agent._deferred_search = MagicMock()
        agent._deferred_fetch = MagicMock()
        EmbodiedAgent._ensure_event_loop(agent)
        ip, tonic = agent._info_processing, agent._tonic
        await ip.close()
        if tonic is not None:
            await tonic.close()
        return agent

    on = asyncio.run(build(True))
    on._deferred_search.set_completion_sink.assert_called_once()
    on._deferred_fetch.set_completion_sink.assert_called_once()
    off = asyncio.run(build(False))
    off._deferred_search.set_completion_sink.assert_not_called()


def test_event_loop_path_starts_mcp_and_memory_worker():
    # MCP とメモリワーカーの起動は run() の中にあり、イベントループの分岐は run() の先頭で
    # return するため到達しなかった。結果 `brave_web_search` が「登録されていない」となり、
    # 検索が 0ms で失敗していた（実機で観測）。CUI/GUI の違いではなく実装の欠落。
    from familiar_agent.agent import EmbodiedAgent

    async def build():
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.event_loop = True
        agent._info_processing = None
        agent._tonic = None
        agent._mcp = MagicMock()
        agent._mcp.is_started = False
        agent._mcp.start = AsyncMock()
        agent._mcp_start_task = None
        agent._memory_worker = MagicMock()
        agent._memory_worker.is_running = False
        agent._memory_worker.start = AsyncMock()
        # MagicMock は未定義属性も返すので、検証対象の実体を明示的に束ねる。
        agent._start_background_services = lambda: EmbodiedAgent._start_background_services(agent)
        EmbodiedAgent._ensure_event_loop(agent)
        await asyncio.sleep(0.02)
        ip, tonic = agent._info_processing, agent._tonic
        await ip.close()
        if tonic is not None:
            await tonic.close()
        return agent

    agent = asyncio.run(build())
    assert agent._mcp.start.called                  # MCP を起こす
    assert agent._memory_worker.start.called        # メモリワーカーも起こす


def test_agent_starts_tonic_only_when_event_loop_is_on():
    # T は EVENT_LOOP on のときだけ立てる。off では従来の経路（GUI の描画ループ）が回すので、
    # 両方が同時に drive を進めないようにする。
    from familiar_agent.agent import EmbodiedAgent

    async def build(event_loop_on: bool):
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.event_loop = event_loop_on
        agent._info_processing = None
        agent._tonic = None
        EmbodiedAgent._ensure_event_loop(agent)
        ip, tonic = agent._info_processing, agent._tonic
        await ip.close()
        if tonic is not None:
            await tonic.close()
        return ip, tonic

    ip_on, tonic_on = asyncio.run(build(True))
    assert ip_on is not None and tonic_on is not None      # on では T を立てる
    ip_off, tonic_off = asyncio.run(build(False))
    assert ip_off is not None and tonic_off is None        # off では立てない


def test_driver_wakes_on_the_affect_queue():
    # 段階3：駆動体は QC と QA の union で起きる。情動（drive 発火）が積まれたら反復が回る。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "ひとりごと"})])])
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        ip.set_output(shown.append)          # 人の発話を待たずに出口を持てる
        ip.start()                           # 駆動体だけ起こす
        ip.push_affect("SEEKING", "何かを知りたい")
        for _ in range(400):
            if shown:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert "".join(shown) == "ひとりごと"


def test_affect_iteration_does_not_send_an_empty_user_message():
    # 情動で起きた反復には人の発話が無い。空の user メッセージを送らず、内的な促しとして渡す。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])

    async def scenario():
        ip = InformationProcessing(a)
        ip.start()
        ip.push_affect("SEEKING", "何かを知りたい")
        for _ in range(400):
            if a.backend.make_user_message.call_count:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    sent = a.backend.make_user_message.call_args.args[0]
    assert sent.strip() != ""
    assert "何かを知りたい" in sent


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


def test_speech_is_held_during_quiet_hours():
    # 静穏時間は「**自分から**話しかけない時間」。判定は配信ゲートに集めるが、掛ける
    # 相手は自発だけにする（以前は起点を区別せず、話しかけられても黙って保留し、翌朝に
    # 届く動きになっていた＝実機で観測）。ここは情動が起点＝自発なので止まる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "ねえ"})])])
    a._in_quiet_hours = MagicMock(return_value=True)

    async def scenario():
        ip = InformationProcessing(a)
        await ip._begin_affect("SEEKING", "なにか気になる")
        await ip.close()

    asyncio.run(scenario())
    a._tts.call.assert_not_awaited()
    a._pending_store.add.assert_called_once()     # 後で話すために積む


def test_full_branch_receives_the_net_actions():
    # 表に載せるだけでは足りず、実際に渡す集合へ入れないとモデルは検索を投げられない
    # （実機で「調べてみるね」と言ったまま何も起きなかった）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a)
    tools = a.backend.stream_turn.call_args.kwargs["tools"]
    assert _SEARCH_DEF in tools and _FETCH_DEF in tools


def test_action_branch_speaks_the_filler_then_dispatches():
    # つなぎの発話は調停が出す（フルLLM を経由しないので速い）。発話したうえで投げる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "使わない"})])])
    a._utility_backend.complete = AsyncMock(return_value=(
        '{"branch":"action","action":"search_deferred",'
        '"query":"今日の天気","text":"調べてみるね"}'))
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        first = await ip.run_iteration("今日の天気を調べて", on_text=shown.append)
        for _ in range(200):
            if a._deferred_search.call.await_count:
                break
            await asyncio.sleep(0.005)
        await ip.close()
        return first

    assert asyncio.run(scenario()) == ""          # 本来の出力はツール投げ
    assert "".join(shown) == "調べてみるね"        # つなぎは即発話
    a.backend.stream_turn.assert_not_awaited()    # フルLLM を起こさない
    assert a._deferred_search.call.await_args.args[1] == {"query": "今日の天気"}


def test_full_branch_keeps_the_tool_when_say_comes_along():
    # フルLLM が「調べてみるね」と検索を同時に返したら、発話をつなぎとして扱い動作も投げる。
    # 以前は say を見つけた時点で閉じ、検索を捨てていた。
    a = _agent(stream_returns=[_turn([
        ToolCall(id="s", name="say", input={"text": "調べてみるね"}),
        ToolCall(id="r", name="search_deferred", input={"query": "今日の天気"}),
    ])])
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        first = await ip.run_iteration("今日の天気を調べて", on_text=shown.append)
        for _ in range(200):
            if a._deferred_search.call.await_count:
                break
            await asyncio.sleep(0.005)
        await ip.close()
        return first

    assert asyncio.run(scenario()) == ""
    assert "".join(shown) == "調べてみるね"        # 発話はつなぎとして出す
    assert a._deferred_search.call.await_count == 1  # 動作は捨てない


def test_net_actions_are_available():
    # deferred（投げっぱなしの外部呼び出し）を投げられなければ、完了キュー経由の
    # 連鎖が意味を持たない。表に2行足すだけで載る。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    assert ip._tools(actions=("search_deferred",)) == [_SEARCH_DEF]
    assert ip._tools(actions=("fetch_deferred",)) == [_FETCH_DEF]


def test_tools_are_selected_by_the_action_set():
    # ツールは「この反復で使える動作の集合」で選ぶ。足すときは表に1行加えるだけで済む
    # ようにしておく（段階3 の次で see・look・search_deferred を載せる）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    assert ip._tools(actions=("say",)) == [_SAY_DEF]
    assert ip._tools(actions=("say", "recall")) == [_SAY_DEF, _RECALL_DEF]
    assert ip._tools(actions=()) == []


def test_unknown_action_is_ignored_not_crashing():
    # 表に無い動作名は黙って落とす（まだ繋いでいない身体を渡そうとしても壊れない）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    assert ip._tools(actions=("say", "walk")) == [_SAY_DEF]


def test_chain_cap_withholds_recall_tool():
    # 連鎖が上限に達した反復では recall を渡さない＝発話を必ず出す（暴走防止）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=2)
    _run_chain(a)
    assert a.backend.stream_turn.call_args_list[0].kwargs["tools"] == [
        _SAY_DEF, _RECALL_DEF, _SEARCH_DEF, _FETCH_DEF]
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


def test_present_ctx_tells_who_is_there_with_confidence():
    # 「誰かが居る」ではなく「誰が居るか」を渡す。確信度も添える（低い認識と高い認識を
    # 同じに扱わない）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a._pmm.presence_status = MagicMock(return_value=[
        {"person_id": "p1", "name": "パパ", "confidence": 0.92, "is_speaker": True},
        {"person_id": "p2", "name": "たいきくん", "confidence": 0.61, "is_speaker": False},
    ])
    _run(a)
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "パパ" in system and "たいきくん" in system
    assert "0.92" in system and "0.61" in system      # 確信度も渡す


def test_present_ctx_says_nobody_is_confirmed_when_no_one_is_recognised():
    # 誰も認識できていないときに黙って空文字を渡すと、自発発話が誰に向けたものか
    # 分からないまま出る。認識できていないことを明示する。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a._pmm.presence_status = MagicMock(return_value=[])
    a._social_presence_permission = MagicMock(return_value=1.0)   # 直近の発話で在席
    _run(a)
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "(present" in system                       # 空文字にしない
    assert "unconfirmed" in system or "確認できていない" in system


def test_datetime_is_injected_into_prompt():
    # 日時が無いと「一昨日」「昨日」を自分で解けず、利用者に日付を聞き返す（実機で観測）。
    # 現行 run() と同じ書式 `(now :datetime "…")` で毎反復渡す。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a)
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "(now :datetime " in system


def test_iteration_context_is_injected_into_prompt():
    # 反復番号と上限をコンテキストで渡す（あと何回で結論すべきかモデルが判断できる）。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ], max_iters=3)
    _run_chain(a)
    systems = ["\n".join(c.kwargs["system"]) for c in a.backend.stream_turn.call_args_list]
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
    # 書いた O は4件（トリガ obs1・open 意図 obs2・完了 obs3・本応答 obs4）。意図と
    # トリガは完了が解決済みなので一括 supersede には載せない（つながりを残すため）。
    # 生き残るのは本応答 obs4 だけで、要約が届いたらそれが supersede する。
    assert a._memory.save_async_with_id.await_count == 4
    assert kwargs["superseded_ids"] == ["obs4"]


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


def test_w_lists_what_has_already_been_looked_up_in_this_chain():
    # 鎖は先頭1件しか生き残らないので、W に載るのは「いちばん新しい完了」だけ。しかも
    # 取得結果は本文が長く（上限8192字）、何を取ったかがその中に埋もれる。実機では
    # 同じ URL を2反復続けて取りに行き、1反復まるごと無駄になった。
    # この求めのために何を調べたかを、短い一覧として別に見せる。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="f", name="fetch_deferred",
                        input={"url": "https://example.com/1hour.html"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
    ])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("今日はどんな天気？")
        ip.push_completion("https://example.com/1hour.html", "時間別の表…")
        for _ in range(200):
            if a.backend.stream_turn.await_count >= 2:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "この求めのために調べたもの" in system
    assert "fetch_deferred「https://example.com/1hour.html」" in system


def test_full_branch_says_a_filler_first_when_thinking_deeply():
    # 正本③ 段5：フルで答えると決めたときは、同じ同期フロー内で二段生成する。軽量LLM が
    # つなぎを即答してから、続けてフルLLM を起こす（1つの work の内部二段＝1反復1出力）。
    # フル生成は effort=high で10秒近くかかり、そのあいだ無音になる。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "本応答"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"full","effort":"high","text":"えーっと"}')
    shown: list[str] = []
    out = _run(a, on_text=shown.append)
    assert out == "本応答"
    assert "えーっと" in "".join(shown)           # つなぎが先に出る
    assert "".join(shown).index("えーっと") < "".join(shown).index("本応答")


def test_full_branch_skips_the_filler_when_the_answer_comes_fast():
    # effort=low のフル生成は実測 0.8〜3.6 秒。速いときに「えーっと」を挟むとテンポが悪い。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "本応答"})])])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"full","effort":"low","text":"えーっと"}')
    shown: list[str] = []
    _run(a, on_text=shown.append)
    assert "えーっと" not in "".join(shown)


def test_filler_is_recorded_so_the_arbiter_knows_it_already_spoke():
    # つなぎは発話なのに O に残らず、次の反復の W に「もう一言伝えた」事実が入らなかった。
    # 調停はそれを知らないので同じことをまた言う（実機で1秒差に同じ文が2回出た）。
    # 抑止ではなく記録で解く：鎖に載れば W に現れ、調停が読んで判断できる。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "マイクラ"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
    ])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"マイクラ",'
                     '"text":"ちょっと調べてみますね"}')

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("マインクラフトってどんなゲーム？")
        head = ip._chain_head_content
        await ip.close()
        return head

    head = asyncio.run(scenario())
    said = [c for c in a._memory.save_async_with_id.call_args_list
            if c.kwargs.get("direction") == "発話" and "調べてみますね" in c.args[0]]
    assert len(said) == 1                      # つなぎが O に1件残る
    # **鎖は進めない。** 進めると直前に届いた完了を押し出し、フルLLM が材料を失う
    # （実機で、想起の結果が W から消えて未回答に終わった）。
    assert "つなぎに言った" not in head


def test_w_lists_what_was_already_said_so_the_next_filler_continues():
    # つなぎを言ったことが W に無いと、同じ言い回しを最初から言い直す。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "サッカー"})]),
        _turn([ToolCall(id="t", name="say", input={"text": "はい"})]),
    ])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"サッカー",'
                     '"text":"ちょっと調べてみますね"}')

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("たいきのサッカーの練習は？")
        ip.push_completion("サッカー", "recall結果テキスト")
        for _ in range(200):
            if a.backend.stream_turn.await_count >= 2:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "すでに相手へ伝えた一言" in system
    assert "ちょっと調べてみますね" in system
    assert "recall結果テキスト" in system     # 完了は押し出されずに残る


def test_quiet_hours_do_not_silence_a_reply_to_a_person():
    # 静穏時間は「自分から話しかけない時間」。話しかけられたのに黙るためのものではない。
    # 起点を区別せず掛けていたため、23時台に話しかけても返事が出ず、保留されて翌朝に
    # 届く動きになっていた（実機で観測）。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    a._in_quiet_hours = MagicMock(return_value=True)
    assert _run(a, utterance="こんばんは") == "やあ"



def test_no_filler_once_the_material_has_arrived():
    # つなぎは待ち時間を埋めるためのもの。結果が届いた反復では待つものが無い。
    # 実機では検索結果の1秒後に「うん、任せてね！」が出て、そこだけ口調が割れた。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "答え"})]),
    ])
    replies = iter([
        '{"branch":"action","action":"recall","query":"q","text":"調べますね"}',
        '{"branch":"full","effort":"high","text":"うん、任せてね！"}',
    ])
    a._utility_backend.complete = AsyncMock(side_effect=lambda *_a, **_k: next(replies))
    shown = _run_chain(a, utterance="調べて")
    assert "調べますね" in shown          # 調べる前のつなぎは出す
    assert "うん、任せてね" not in shown  # 届いたあとは出さない
    assert "答え" in shown


def test_the_arbiter_sees_what_was_already_looked_up():
    """2回目の反復で、調停に「この求めのために調べたもの」が届いていること。

    実機で同じ `recall` を4反復続けて投げた（語は MD5 まで一致）。W に一覧は出している
    はずだが、**届いたのに従わないのか、そもそも届いていないのか**を区別する手立てが
    無かった。ここで機構として確かめる。
    """
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"直近の天気","text":"調べますね"}')
    _run_chain(a, utterance="どこの天気？")

    prompts = [c.args[0] for c in a._utility_backend.complete.call_args_list]
    assert len(prompts) >= 2, "2反復目まで回っていない"
    assert "この求めのために調べたもの" in prompts[1]
    assert "直近の天気" in prompts[1]
