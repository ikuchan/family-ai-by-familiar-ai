"""主LLM を投げたあとと、主LLM の答えが届いたときの版は、裏で書く（出-bz-ろ・2026-10-13・本人の決定）。

版を書くには埋め込みと新しさの測りが要り、約 6,800 字で 0.8 秒かかる。10/13 00:35 には、主LLM を投げたあとの版で
駆動体が 0.82 秒止まり、主LLM の答えが届いたときの版で、答えの声の前に 0.80 秒かかった。どちらも、書き終わりを
待っているものは無い。調べものの結果が届いたときの版は、直後の想起で使うので今のまま待つ（本人）。

裏で書いても版の鎖の順番は崩さない（1 本の鍵で順に書く）。求めを閉じるときは、裏で書いている版を待ってから
やりとりの関係をまとめる（待たないと、閉じたあとに版の記録が届き、次の求めの欄に混ざる）。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from unittest.mock import MagicMock

from familiar_agent.loop import iteration
from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent


def _ip(*, write_takes: float = 0.2):
    ip = InformationProcessing(_agent(stream_returns=[]))
    written: list[str] = []
    ids = iter(f"v-{i}" for i in range(1, 99))

    async def write(mi, **kw):
        await asyncio.sleep(write_takes)
        written.append(mi.content)
        return next(ids)

    ip._agent._oif.write = write
    ip._agent._oif.supersede = MagicMock()
    ip._req.request_id = "req-1"
    ip._req.request_text = "明日の天気は？"
    return ip, written


def test_a_later_write_does_not_wait():
    ip, written = _ip()
    took: list[float] = []

    async def go():
        loop = asyncio.get_running_loop()
        started = loop.time()
        ip._write_version_later()
        took.append(loop.time() - started)
        assert written == []
        await ip._settle_versions()
        await ip.close()

    asyncio.run(go())
    assert took[0] < 0.05
    assert written == ["「明日の天気は？」と聞かれた"]
    assert ip._req.live_version_id == "v-1"


def test_versions_are_written_in_order():
    """裏で書いている版のあとに待って書く版が来ても、鎖の順番は崩れない。"""
    ip, written = _ip()

    async def go():
        ip._write_version_later()
        ip._req.lookups.append(
            Lookup(index=1, action="search_deferred", query="天気", generation=0)
        )
        await ip._write_version()
        await ip.close()

    asyncio.run(go())
    assert written[0] == "「明日の天気は？」と聞かれた"
    assert "1番：search_deferred「天気」を起動中" in written[1]
    ip._agent._oif.supersede.assert_called_once_with("v-1", "v-2", kind="改訂")
    assert ip._req.live_version_id == "v-2"


def _intake(ip, trigger):
    ip._drained_completions.append(trigger)

    async def go():
        started = asyncio.get_running_loop().time()
        await ip._intake()
        took = asyncio.get_running_loop().time() - started
        await ip._settle_versions()
        await ip.close()
        return took

    return asyncio.run(go())


def test_the_answer_of_the_main_llm_is_versioned_in_the_background():
    ip, written = _ip()
    ip._req.lookups.append(Lookup(index=1, action="主LLM", query="主LLM1", generation=0))
    took = _intake(ip, Trigger(kind="決定", query="主LLM1", index=1, decision=MagicMock()))
    assert took < 0.1
    assert written and "1番：考えて答えた" in written[-1]


def test_a_lookup_result_is_versioned_before_recall():
    """**反証の側**：調べものの結果が届いたときは、想起で使うので書き終わるまで待つ（本人）。"""
    ip, written = _ip()
    ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="天気", generation=0))
    took = _intake(ip, Trigger(kind="完了", query="天気", result="晴れ", index=1))
    assert took >= 0.2
    assert "の結果が届いた：晴れ" in written[-1]


def test_finish_waits_for_the_version():
    ip, written = _ip()
    seen: list = []

    def close_exchange():
        seen.append(list(written))
        return None

    ip._close_exchange = close_exchange  # type: ignore[method-assign]

    async def go():
        ip._write_version_later()
        with contextlib.suppress(Exception):
            await ip._finish("", [], "沈黙")
        await ip.close()

    asyncio.run(go())
    # 裏の版を書き終えてから閉じる。最後に、閉じるときの最終版（記-t）が版の鎖の最後に付く。
    from familiar_agent.core.final_version import is_final

    assert seen and seen[0][0] == "「明日の天気は？」と聞かれた"
    assert len(seen[0]) == 2 and is_final(seen[0][1])


def test_the_iteration_writes_the_thinking_version_later():
    src = inspect.getsource(iteration.Iteration._throw_main_llm)
    assert "ip._write_version_later()" in src and "await ip._write_version()" not in src
