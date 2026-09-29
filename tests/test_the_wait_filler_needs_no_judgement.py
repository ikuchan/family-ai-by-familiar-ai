"""待ちの知らせで起きた反復は、分岐を判定せず、つなぎだけを書かせる（出-aq 段 6・2026-09-29）。

以前は、つなぎ 1 つのために反復をまるごと回し、Jev が選んだ分岐しだいで言うことが変わった——light なら
返事（内容に触れる）を、full で深さが low なら何も書かれず 5 秒たっても黙った。**つなぎを出すのは
待たせている事実であって、分岐ではない。** 軽量LLM に `filler` と `trash`（捨て場）だけを書かせ、何を
待っているか（調べものか、考えている最中か）は機械が渡す。続き先の判定と依頼の読み取りは回さない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests._arbiter_fakes import jev_says, prompt_of, writer_says
from tests.test_event_loop import _agent


def _progress(action: str, *, writer, lookups=True):
    a = _agent(stream_returns=[])
    a._jev = jev_says("light")  # 聞かれたら「返事で答える」と言う Jev。聞かれてはいけない
    a._utility_backend = writer
    shown: list[str] = []

    async def scenario():
        ip = InformationProcessing(a)
        ip._wake_window().open(time.monotonic())  # 入口を通った会話（出-as 段 4）
        ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
        ip.set_output(shown.append)
        ip._judge_follows = AsyncMock()  # type: ignore[method-assign]
        ip._apply_requests = AsyncMock()  # type: ignore[method-assign]
        ip._req.utterance = "明日の天気は？"
        if lookups:
            ip._req.lookups = [Lookup(index=1, action=action, query="明日の天気", generation=0)]
        ip._drained_completions.append(Trigger(kind="進捗", query="明日の天気"))
        await ip._iterate()
        await ip.close()
        return ip

    ip = asyncio.run(scenario())
    return a, ip, "".join(shown)


def test_the_filler_is_written_without_asking_jev():
    writer = writer_says({"filler": "いま調べていますね", "trash": ""})
    a, ip, shown = _progress("search_deferred", writer=writer)
    assert "いま調べていますね" in shown
    a._jev.ask.assert_not_awaited()


def test_no_follow_up_or_request_reading_for_a_filler():
    _, ip, _ = _progress("search_deferred", writer=writer_says({"filler": "もう少しです"}))
    ip._judge_follows.assert_not_awaited()
    ip._apply_requests.assert_not_awaited()


def _decided(writer) -> str:
    """書かせる文のうち「何をするか」の行だけ。決まり文句にも「調べている」「考えている」が入るので、そこは見ない。"""
    return next(ln for ln in prompt_of(writer).splitlines() if ln.startswith("次にすることは"))


def test_the_writer_is_told_a_lookup_is_running():
    writer = writer_says({"filler": "もう少しです"})
    _progress("search_deferred", writer=writer)
    assert "調べ" in _decided(writer) and "考え" not in _decided(writer)


def test_the_writer_is_told_it_is_still_thinking():
    writer = writer_says({"filler": "もう少しです"})
    _progress("主LLM", writer=writer)
    assert "考え" in _decided(writer) and "調べ" not in _decided(writer)


def test_only_the_filler_and_its_trash_are_asked_for():
    writer = writer_says({"filler": "もう少しです"})
    _progress("search_deferred", writer=writer)
    shape = prompt_of(writer).rsplit("\n", 2)[-2]
    assert '"filler"' in shape and '"trash"' in shape
    assert '"text"' not in shape and '"query"' not in shape


def test_nothing_is_said_once_the_answer_is_no_longer_awaited():
    writer = writer_says({"filler": "もう少しです"})
    _, _, shown = _progress("search_deferred", writer=writer, lookups=False)
    assert shown == ""
    writer.complete.assert_not_awaited()


def test_a_writer_that_fails_leaves_silence():
    _, _, shown = _progress("search_deferred", writer=writer_says(raises=RuntimeError("down")))
    assert shown == ""
