"""一つのターンの記録を、やりとりの関係として順序つきで残す（段 3）。

問いと答えを別々の記憶として置くと、対として想起されない（`設計方針_MI間の関係`）。
起点と版と見た結果と答えの逐語と会話要約を、一つの関係の項として**順序つきで**並べる。

項は時間差で揃う（起点はターン頭、答えは反復の終わり、要約は背景タスク）。全部が揃う
のは `_run_post_response_pipeline` の中なので、関係はそこで一度に書く。
"""

from __future__ import annotations

import asyncio
from familiar_agent.io.oif import OIF
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.backends import ToolCall
from tests.test_event_loop import _WAIT_TICKS, _agent, _exchange_members, _run, _turn
from familiar_agent.loop.event_loop import InformationProcessing


def _exchange(a):
    """閉じるときに同期で書かれたやりとりの項（2026-09-13 まで背景へ `exchange=` で渡していた）。"""
    return _exchange_members(a)


def test_the_turn_hands_over_its_records_in_order():
    """起点が先、答えが後。順序は関係の `position` になる。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})])])
    _run(a, utterance="今日の天気は？")
    # **件数は固定しない。** 環-h で主LLM の投げと返りにも版が書かれ、あいだが増えた。
    # 守るのは順序——先頭が起点、末尾が答え、あいだは求めの版である。
    roles = [r for _i, r in _exchange(a)]
    assert roles[0] == "起点"
    assert roles[-1] == "答え"
    assert set(roles[1:-1]) == {"版"}


def test_a_silent_turn_hands_over_no_answer():
    """黙ったターンには答えの項ができない。常に作っていれば、ここが落ちる。"""
    a = _agent(stream_returns=[_turn([], text="")])
    _run(a, utterance="…")
    roles = [r for _, r in _exchange(a)]
    assert "答え" not in roles


def _pipeline_agent():
    agent = MagicMock()
    agent.config = MagicMock()
    agent._emotion_for_turn = AsyncMock(return_value=(None, 0.0, "neutral"))
    agent._update_mood = MagicMock()
    agent._summarize_exchange = AsyncMock(return_value="summary")
    agent._maybe_update_self_narrative = AsyncMock()
    agent._maybe_adapt_values = AsyncMock()
    agent._maybe_discharge_satisfied_drives = AsyncMock()
    agent._active_memory = MagicMock(return_value=agent._memory)
    agent._memory.save_async_with_id = AsyncMock(return_value=("conv-1", True))
    agent._memory.link = MagicMock(return_value=1)  # 関係は口を通る（環-e-い）
    agent._conversation_perspective = MagicMock(return_value={"writer_id": "話者"})
    # 関係も書き込みも OIF を通る（環-e-い）。**口は本物・内側の記憶だけ偽物**に
    # すれば、記憶への検証がそのまま効く。
    agent._oif = OIF(agent._memory)
    return agent


def test_the_summary_is_appended_as_the_last_member():
    """会話要約は最後に来る。背景で遅れて作られるが、位置は末尾で決まっている。

    やりとりの関係そのものは反復を閉じるときに同期で書かれる（2026-09-13）。背景は
    その関係の id を受け取り、要約を**末尾へ足す**だけである。
    """
    agent = _pipeline_agent()

    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="今日の天気は？",
            final_text="晴れだよ",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
            is_desire_turn=False,
            desires=None,
            exchange_id=7,
        )
    )

    rid, members = agent._memory.extend.call_args.args  # extend(relation_id, members)
    assert rid == 7 and members == [("conv-1", "要約", None)]
    # 背景では関係を**新たに作らない**。
    assert not [c for c in agent._memory.link.call_args_list if c.args[0] == "やりとり"]


def test_no_relation_is_written_when_the_turn_left_nothing():
    """項が要約だけなら、やりとりとは呼べない。関係を書かない。"""
    agent = _pipeline_agent()

    asyncio.run(
        EmbodiedAgent._run_post_response_pipeline(
            agent,
            user_input="",
            final_text="ひとりごと",
            camera_used=False,
            camera_image=None,
            observation_action_name=None,
            observation_action_input=None,
            companion_mood="engaged",
            is_desire_turn=False,
            desires=None,
            exchange_id=None,
        )
    )

    agent._memory.link.assert_not_called()
    agent._memory.extend.assert_not_called()


def test_an_interrupted_turn_does_not_leak_into_the_next_one():
    """話しかけられて調べかけを打ち切ったら、そこで一つのやりとりが閉じる。

    閉じないと、打ち切られた問いと新しい問いが**一つのやりとり**に入る（起点が2つ）。
    母集合への持ち越しは別で、打ち切りの記録は次のターンの WR にも載り続ける。
    """
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "明日は晴れだよ"})]),
        ]
    )

    # 調べものが返らないようにする。返ると1つめの求めがそのまま答えてしまい、
    # 「調べかけを打ち切る」場面にならない。
    never = asyncio.Event()

    async def hang(*_args, **_kwargs):
        # `exclude_ids` も来るので**すべて受ける**。取りこぼすと TypeError が
        # 「recall を実行できなかった」完了として畳まれ、止まらずに先へ進む。
        await never.wait()
        return ("届かない", None)

    a._memory_tool.call = AsyncMock(side_effect=hang)

    async def scenario():
        ip = InformationProcessing(a)
        await ip.push_utterance("昨日の天気覚えてる？")
        # 環-h で主LLM は投げっぱなしになった。調べかけになるまで待つ。
        for _ in range(_WAIT_TICKS):
            if a._memory_tool.call.called:
                break
            await asyncio.sleep(0.005)
        # 調べかけの途中で話しかける。
        await ip.push_utterance("それより明日の予定は？")
        for _ in range(_WAIT_TICKS):
            if a._run_post_response_pipeline.called:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())

    # 打ち切りで閉じたやりとりに、新しい問いは入っていない。
    aborted = a._memory.link.call_args.args[1]  # link(kind, members)
    assert [r for _, r, _ in aborted].count("起点") == 1, aborted

    # 続くターンのやりとりにも、起点は1つだけ。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert [r for _, r in _exchange_members(a)].count("起点") == 1, _exchange_members(a)
    # 母集合へは、打ち切りの分も持ち越して渡る。
    assert "obs1" in kwargs["extra_cooccurring_ids"]


def test_the_exchange_is_linked_before_the_background_pipeline_starts():
    """やりとりの関係は反復を閉じるときに**同期で**書く（2026-09-13 実機・F）。

    背景の要約待ちで書いていたため、閉じた 0.1 秒後に起きた入室の反復が
    `OIF exchanges → 0件` を引き、直近のやりとりが空のまま調停へ渡った（関係が書かれた
    のは 2.4 秒後）。こうきと話した直後に「おかえり、こうき！」と挨拶した。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はい"})])])
    a._memory.link = MagicMock(return_value=55)
    _run(a, utterance="お話できる？")

    exchanges = [c.args[1] for c in a._memory.link.call_args_list if c.args[0] == "やりとり"]
    assert exchanges, "反復を閉じるときにやりとりの関係が書かれていない"
    roles = [r for _i, r, _p in exchanges[-1]]
    assert "起点" in roles and "答え" in roles, roles
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["exchange_id"] == 55
    assert "exchange" not in kwargs
