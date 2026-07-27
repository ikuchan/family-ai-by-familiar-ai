"""ループが作った記録を、拡散想起の母集合（WR）へ載せる。

拡散想起は WR の共起をたどる。**逐語（自分が答えた記録）はどの WR にも載っておらず**、
辿り着く辺が無かった（実機で、逐語の WR 掲載数が 0 だった）。会話要約が2秒後に逐語を
supersede するので、想起の4軸からも引けない。拡散想起は `superseded_by` を条件にして
いないので**閉じた記録でも引ける**が、母集合に無ければ届かない。

載せるのは**意図・完了・中断・逐語**。つなぎは載せない（中身が無く、共起として育てる
価値がない）。中断はその求めで閉じるが、次の求めの WR に載る（打ち切った調査と言い直した
問いの共起は、たどる価値がある）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backend import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _run, _run_chain, _turn


def _extra_wr_ids(a):
    _, kwargs = a._run_post_response_pipeline.call_args
    return list(kwargs.get("extra_wr_ids") or [])


def test_the_answer_is_put_into_the_diffuse_pool():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})])])
    _run(a, utterance="今日の天気は？")
    # obs1=トリガ / obs2=逐語。載るのは逐語だけ（トリガは W と別経路で載る）。
    assert _extra_wr_ids(a) == ["obs2"]


def test_intent_and_completion_are_put_into_the_diffuse_pool():
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    _run_chain(a, utterance="調べて")
    # obs2=意図 / obs3=完了 / obs4=逐語。
    assert _extra_wr_ids(a) == ["obs2", "obs3", "obs4"]


def test_the_filler_is_not_put_into_the_pool():
    from unittest.mock import AsyncMock

    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"q","text":"調べますね"}')
    _run_chain(a, utterance="調べて")
    written = {c.args[0]: c for c in a._memory.save_async_with_id.call_args_list}
    filler_ids = [i for i, (content, _) in enumerate(written.items()) if "つなぎに言った" in content]
    assert filler_ids, "つなぎが書かれていない（前提が崩れている）"
    # つなぎの id は、書かれた順で分かる。母集合には入っていないこと。
    assert all("つなぎ" not in c.args[0]
               for c in a._memory.save_async_with_id.call_args_list
               if c.args[0] in _extra_wr_ids(a))


def test_an_aborted_investigation_is_carried_to_the_next_pool():
    # 中断はその求めで閉じるが、次の求めの WR に載る。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._parent_id = "obs-parent"
    ip._in_flight_lookups = [("recall", "前の調査")]
    asyncio.run(ip._abort_investigation())
    assert ip._wr_ids, "中断の記録が母集合へ控えられていない"
