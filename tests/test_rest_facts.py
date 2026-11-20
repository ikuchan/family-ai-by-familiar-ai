"""記-ad：REST 内省・層 1 の⑤「事実を抜く」（`loop/rest_facts`）。

抜き待ちの本文を束にして軽量LLM に聞き、返りを鍵の文にして書く。記憶は偽物、LLM も偽物（DB 不要）。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.fact_keys import Fact
from familiar_agent.loop import rest_facts as rf
from familiar_agent.store.fact_keys import FactSource


class _Store:
    def __init__(self, n: int):
        self.sources = [FactSource(f"本文{i}", (f"id{i}",)) for i in range(n)]
        self.written: list = []
        self.asked: list[int] = []

    def fact_pending(self, limit):
        self.asked.append(limit)
        return self.sources[:limit]

    ready = True

    def fact_ready(self):
        return self.ready

    def write_fact_keys(self, sources, facts, *, model):
        self.written.append((list(sources), facts, model))
        done = [f for f in facts if f is not None]
        return len(done), sum(2 * len(f) for f in done)


def _reply(n, with_fact=(0,)):
    return json.dumps(
        [
            {"i": i, "facts": [{"s": "s", "r": f"r{i}", "a": "a"}] if i in with_fact else []}
            for i in range(n)
        ]
    )


def _backend(*replies):
    b = MagicMock()
    b.model = "claude-haiku-5-5"
    b.complete = AsyncMock(side_effect=list(replies))
    return b


def test_sources_are_asked_in_batches_and_written_with_the_model_name():
    store = _Store(10)
    b = _backend(_reply(8), _reply(2))
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    assert store.asked == [400]
    assert b.complete.await_count == 2
    assert [len(s) for s, _, _ in store.written] == [8, 2]
    assert store.written[0][1][0] == [Fact("s", "r0", "a")] and store.written[0][1][1] == []
    assert store.written[0][2] == "claude-haiku-5-5"
    assert (r.sources, r.done, r.facts, r.failed_batches) == (10, 10, 2, 0)


def test_an_unreadable_reply_leaves_that_batch_pending_and_goes_on():
    store = _Store(16)
    b = _backend("読めない", _reply(8))
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    assert len(store.written) == 1 and store.written[0][0] == store.sources[8:]
    assert r.failed_batches == 1


def test_three_failures_in_a_row_stop_the_night():
    """残高切れや障害で LLM が返さないとき、同じ失敗を束の数だけ重ねない。"""
    store = _Store(80)
    b = _backend(*[""] * 10)
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    assert b.complete.await_count == 3
    assert r.failed_batches == 3 and not store.written


def test_when_ruri_fails_the_batch_counts_as_failed():
    store = _Store(8)
    store.write_fact_keys = lambda *a, **k: None
    r = asyncio.run(rf.extract_facts(store, _backend(_reply(8)), limit=400, batch=8))
    assert r.failed_batches == 1 and r.facts == 0


def test_nothing_pending_calls_no_llm():
    store = _Store(0)
    b = _backend()
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    assert b.complete.await_count == 0 and r.sources == 0


def test_the_prompt_holds_the_bodies_and_the_llm_contents_stay_out_of_info(caplog):
    import logging

    store = _Store(2)
    b = _backend(_reply(2))
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.rest_facts"):
        asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    prompt = b.complete.await_args.args[0]
    assert "[0] 本文0" in prompt and "[1] 本文1" in prompt
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert info and all("本文0" not in m and "本文1" not in m for m in info)


def test_the_rest_pass_runs_it_after_the_experience_links():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_links import Outcome

    order: list[str] = []
    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="obs1")
    agent._observation_perspective = MagicMock(return_value={})

    async def links(_agent):
        order.append("経験繋がり")
        return Outcome(written=0, replaced=0, removed=0)

    async def facts(_agent):
        order.append("事実")
        return rf.FactsResult(sources=5, done=4, facts=7, keys=14, failed_batches=0)

    with (
        patch("familiar_agent.loop.rest.link_experiences", new=links),
        patch("familiar_agent.loop.rest.extract_facts_tonight", new=facts),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["経験繋がり", "事実"]
    assert "抜き待ち 5 種のうち 4 種から事実を 7 個抜いた" in content


def test_workers_ask_several_batches_at_once_and_keep_the_order_of_writes():
    """埋める道具は束を並べて聞く（4 つずつ）。書くのは束の順のまま。"""
    store = _Store(24)
    running = {"now": 0, "max": 0}

    async def complete(prompt, max_tokens):
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        await asyncio.sleep(0.01)
        running["now"] -= 1
        return _reply(8)

    b = MagicMock()
    b.model = "m"
    b.complete = complete
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8, workers=3, timeout=5))
    assert running["max"] == 3
    assert [s[0].obs_ids for s, _, _ in store.written] == [("id0",), ("id8",), ("id16",)]
    assert r.facts == 3


def test_a_batch_that_raises_counts_as_failed_and_the_others_are_written():
    store = _Store(16)
    calls = {"n": 0}

    async def complete(prompt, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("落ちた")
        return _reply(8)

    b = MagicMock()
    b.model = "m"
    b.complete = complete
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8, workers=2))
    assert r.failed_batches == 1 and len(store.written) == 1


def test_tonight_skips_when_only_the_main_llm_is_there():
    """軽量LLM が無いと `_utility_backend` は主LLM になる。高いモデルで毎晩 400 種は抜かない。"""
    agent = MagicMock()
    agent._utility_backend = agent.backend
    r = asyncio.run(rf.extract_facts_tonight(agent))
    assert r.sources == 0 and r.skipped == "軽量LLM が無い"
    agent._oif.fact_pending.assert_not_called()


def test_when_ruri_cannot_embed_no_llm_is_called():
    store = _Store(8)
    store.ready = False
    b = _backend(_reply(8))
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8))
    assert b.complete.await_count == 0 and r.done == 0 and r.sources == 8
    assert r.skipped == "Ruri が使えない"


def test_a_write_that_raises_counts_as_failed_and_the_night_goes_on():
    store = _Store(16)
    calls = {"n": 0}
    real = store.write_fact_keys

    def write(sources, facts, *, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("DB が落ちた")
        return real(sources, facts, model=model)

    store.write_fact_keys = write
    r = asyncio.run(rf.extract_facts(store, _backend(_reply(8), _reply(8)), limit=400, batch=8))
    assert r.failed_batches == 1 and r.done == 8


def test_replies_already_paid_for_in_a_group_are_written_even_after_three_failures():
    store = _Store(32)
    b = _backend("", "", "", _reply(8))
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8, workers=4))
    assert len(store.written) == 1 and r.failed_batches == 3


def test_a_call_that_hangs_times_out_and_counts_as_failed():
    store = _Store(8)

    async def complete(prompt, max_tokens):
        await asyncio.sleep(10)

    b = MagicMock()
    b.model = "m"
    b.complete = complete
    r = asyncio.run(rf.extract_facts(store, b, limit=400, batch=8, timeout=0.05))
    assert r.failed_batches == 1 and not store.written


def test_only_the_records_written_count_as_done():
    store = _Store(8)
    r = asyncio.run(
        rf.extract_facts(store, _backend('[{"i": 0, "facts": []}]'), limit=400, batch=8)
    )
    assert r.done == 1 and r.failed_batches == 0


def test_the_rest_pass_says_why_it_did_nothing():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass

    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="obs1")
    agent._observation_perspective = MagicMock(return_value={})

    async def facts(_agent):
        return rf.FactsResult(
            sources=0, done=0, facts=0, keys=0, failed_batches=0, skipped="軽量LLM が無い"
        )

    with patch("familiar_agent.loop.rest.extract_facts_tonight", new=facts):
        content = asyncio.run(run_rest_pass(agent))
    assert "事実を抜かなかった（軽量LLM が無い）" in content
