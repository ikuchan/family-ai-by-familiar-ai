"""ループが作った記録を、拡散想起の母集合（WR）へ載せる。

拡散想起は WR の共起をたどる。**逐語（自分が答えた記録）はどの WR にも載っておらず**、
辿り着く辺が無かった（実機で、逐語の WR 掲載数が 0 だった）。

載せるのは**起点・意図・完了・中断・逐語**。つなぎは載せない（中身が無く、共起として
育てる価値がない）。中断はその求めで閉じるが、次の求めの WR に載る（打ち切った調査と
言い直した問いの共起は、たどる価値がある）。

**起点も載せる**（段 3）。載せないと、問いだけが母集合に入らず、問いから答えへ辿れない。
同じ並びがやりとりの関係の項にもなる（`設計方針_MI間の関係`）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from tests.test_event_loop import _agent, _run, _run_chain, _turn


def _written_ids(a):
    """このターンで O へ書いた記録の id を、書いた順に並べる（土台が振る obs1, obs2, …）。"""
    return [f"obs{i}" for i in range(1, a._memory.save_async_with_id.await_count + 1)]


def _extra_cooccurring_ids(a):
    _, kwargs = a._run_post_response_pipeline.call_args
    return list(kwargs.get("extra_cooccurring_ids") or [])


def test_the_answer_is_put_into_the_diffuse_pool():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})])])
    _run(a, utterance="今日の天気は？")
    # **起点も載せる**（段 3）。載せないと、問いだけが母集合に入らず、拡散想起が問いから
    # 答えへ辿れない。**件数は固定しない**——環-h で主LLM の投げと返りにも版が書かれ、
    # 増えた。守るのは「このターンで書いた記録が残らず載る」ことである。
    assert _extra_cooccurring_ids(a) == _written_ids(a)


def test_intent_and_completion_are_put_into_the_diffuse_pool():
    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
        ]
    )
    _run_chain(a, utterance="調べて")
    assert _extra_cooccurring_ids(a) == _written_ids(a)


def test_the_filler_is_written_and_pooled():
    from unittest.mock import AsyncMock

    a = _agent(
        stream_returns=[
            _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
            _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
        ]
    )
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"q","text":"調べますね"}'
    )
    _run_chain(a, utterance="調べて")
    # つなぎは O へ書く（段 4）。**ただし母集合には載せる。** 想起から外すのは役割が
    # 担っており、拡散想起は役割を見ない。載せないと、聞こえた一言へ辿り着く辺が無い。
    written = [c.args[0] for c in a._memory.save_async_with_id.call_args_list]
    assert any(str(t).startswith("つなぎに言った：") for t in written), "つなぎを書いていない"


def test_an_aborted_investigation_is_carried_to_the_next_pool():
    # 中断はその求めで閉じるが、次の求めの WR に載る。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._req.request_id = "obs-parent"
    ip._req.lookups = [Lookup(index=1, action="recall", query="前の調査", generation=0)]
    asyncio.run(ip._abort_lookups())
    assert ip._req.turn_records, "中断の記録が母集合へ控えられていない"
