"""待たせているあいだ、つなぎ（待ってもらう一言）を言わない。見張りは窓を延ばすだけ（出-cl 段 2・2026-10-15・本人の決定ア）。

考えているあいだは合図の声と待ちの電子音で分かるので、軽量LLM が書くつなぎはやめた。つなぎの不具合（出-cg：見張りが
知らせない、10/14 23:03：求めが閉じた後に鳴った）は、直す対象ごと無くなる。窓（10 秒）が待ちのあいだに切れると、出来上がった
答えが独り言になるので、会話の求めで待たせているあいだは見張りが窓を延ばし続ける（2026-10-07 本人の決定）。
"""

from __future__ import annotations

import asyncio
import importlib
import time

import pytest

from familiar_agent.config import AgentConfig
from familiar_agent.loop import event_loop
from familiar_agent.loop.arbiter import Arbiter
from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from familiar_agent.loop.request import Request
from tests.test_event_loop import _agent


@pytest.fixture(autouse=True)
def _fast_watch(monkeypatch):
    monkeypatch.setattr(event_loop, "WAIT_WATCH_TICK_SEC", 0.01, raising=False)


def _ip(kind: str, action: str = "search_deferred"):
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.trigger_kind = kind
    ip._req.request_id = "req-1"
    ip._req.lookups = [Lookup(index=1, action=action, query="q", generation=0)]
    return ip


def _watch(ip, *, run_for: float = 0.08):
    w = ip._wake_window()
    w.open(time.monotonic() - 9.95)  # あと 0.05 秒で閉じる窓

    async def go():
        ip._ensure_wait_watch()
        await asyncio.sleep(run_for)
        ip._stop_wait_watch()

    asyncio.run(go())
    return w, time.monotonic()


@pytest.mark.parametrize("action", ["search_deferred", "主LLM"])
def test_a_long_wait_queues_no_filler(action):
    ip = _ip("発話", action)
    _watch(ip)
    kinds = []
    while not ip._triggers.empty():
        kinds.append(ip._triggers.get_nowait().kind)
    assert kinds == []


def test_the_window_stays_open_while_a_conversation_waits():
    """**反証の側**：見張りは残り、窓を延ばし続ける。"""
    w, now = _watch(_ip("発話"))
    assert w.is_open(now)
    assert not w.is_open(now + 10.5)  # 待ちが終われば最後に延ばしたところから 10 秒で閉じる


def test_affect_does_not_keep_the_window():
    w, now = _watch(_ip("情動"))
    assert not w.is_open(now)


def test_only_one_watch_per_request():
    ip = _ip("発話")

    async def go():
        ip._ensure_wait_watch()
        first = ip._wait_watch
        ip._ensure_wait_watch()  # 2 件目の調べもの・主LLM でも増やさない
        same = ip._wait_watch is first
        ip._stop_wait_watch()
        return same

    assert asyncio.run(go())


def test_an_aborted_request_stops_extending_the_window():
    ip = _ip("発話")
    w = ip._wake_window()
    w.open(time.monotonic() - 9.95)

    async def go():
        ip._ensure_wait_watch()
        ip._request_generation += 1  # 打ち切られた
        await asyncio.sleep(0.08)
        ip._stop_wait_watch()

    asyncio.run(go())
    assert not w.is_open(time.monotonic())


def test_the_filler_machinery_is_gone():
    for name in ("_say_waiting_filler", "_write_and_say_filler"):
        assert not hasattr(InformationProcessing, name), name
    assert not hasattr(Arbiter, "write_filler")
    req = Request()
    for name in (
        "filler_pending",
        "last_progress_at",
        "cue_words",
        "last_sound_ended_at",
        "heard_at",
        "answering",
    ):
        assert not hasattr(req, name), name
    cfg = AgentConfig()
    for name in ("lookup_slow_seconds", "wait_filler_gap_seconds"):
        assert not hasattr(cfg, name), name
    with pytest.raises(ImportError):
        importlib.import_module("familiar_agent.core.filler_length")
