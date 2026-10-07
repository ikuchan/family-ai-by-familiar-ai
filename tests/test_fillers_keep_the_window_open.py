"""待たせている間はつなぎを繰り返し、窓を保つ（出-au 段 2・2026-09-27・`設計方針_判定の段` §2.3）。

窓は 10 秒（2026-10-07 から・以前は 30 秒）。調べものか主LLM を待つあいだに窓が切れると、出来上がった答えは独り言になって消える。
**会話の求めでは**、待たせている時間が 5 秒（`lookup_slow_seconds`）を超えたら 1 回、その後は 20 秒
（`wait_filler_repeat_seconds`）ごとに「進捗」を積み、つなぎを出す（つなぎは窓を延ばす）。主LLM の待ちにも付ける。
情動と機器の求めには見張りを立てない（出-aq 段 6・つなぎは会話の求めだけ。待たせている相手が居ない）。

- 見張りは**求めごとに 1 本**。求めが閉じる・打ち切られると止まる。
- 「進捗」で起きた反復は反復の上限に数えない（数えると繰り返しで上限に届き、答える前に打ち切りになる）。
- 本物の結果と「進捗」が同じ反復に届いたら、結果に答える（つなぎだけ言って返ると、答える反復が二度と起きない）。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent


def _ip(kind: str, action: str, *, first=0.01, every=0.02):
    a = _agent(stream_returns=[])
    a.config.lookup_slow_seconds = first
    a.config.wait_filler_repeat_seconds = every
    ip = InformationProcessing(a)
    ip._req.trigger_kind = kind
    ip._req.request_id = "req-1"
    ip._req.lookups = [Lookup(index=1, action=action, query="q", generation=0)]
    return ip


def _progress_count(ip, *, run_for: float) -> int:
    async def go():
        ip._ensure_wait_watch()
        await asyncio.sleep(run_for)
        ip._stop_wait_watch()
        n = 0
        while not ip._triggers.empty():
            n += ip._triggers.get_nowait().kind == "進捗"
        return n

    return asyncio.run(go())


def test_the_repeat_interval_comes_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        cfg = AgentConfig()
        assert (cfg.lookup_slow_seconds, cfg.wait_filler_repeat_seconds) == (5.0, 20.0)


def test_a_conversation_keeps_hearing_progress_while_waiting():
    ip = _ip("発話", "search_deferred")
    assert _progress_count(ip, run_for=0.1) >= 3  # 0.01 で 1 回、その後 0.02 ごと


def test_the_main_llm_wait_is_watched_in_a_conversation():
    ip = _ip("発話", "主LLM")
    assert _progress_count(ip, run_for=0.06) >= 2


def test_affect_and_devices_hear_no_progress():
    # 以前は情動の調べものが遅いと 1 回積んだ。つなぎを会話の求めだけにしたので、積まない（出-aq 段 6）。
    for kind in ("情動", "機器"):
        assert _progress_count(_ip(kind, "search_deferred"), run_for=0.1) == 0
        assert _progress_count(_ip(kind, "主LLM"), run_for=0.1) == 0


def test_nothing_in_flight_means_no_progress():
    ip = _ip("発話", "search_deferred")
    ip._req.lookups = []
    assert _progress_count(ip, run_for=0.05) == 0


def test_an_aborted_request_stops_the_watch():
    ip = _ip("発話", "search_deferred")

    async def go():
        ip._ensure_wait_watch()
        ip._request_generation += 1  # 打ち切られた
        await asyncio.sleep(0.06)
        ip._stop_wait_watch()
        return ip._triggers.empty()

    assert asyncio.run(go())


def test_only_one_watch_per_request():
    ip = _ip("発話", "search_deferred", every=0.05)

    async def go():
        ip._ensure_wait_watch()
        ip._ensure_wait_watch()  # 2 件目の調べもの・主LLM でも増やさない
        await asyncio.sleep(0.03)
        ip._stop_wait_watch()
        n = 0
        while not ip._triggers.empty():
            n += 1
            ip._triggers.get_nowait()
        return n

    assert asyncio.run(go()) == 1


# ── 反復の数え方 ───────────────────────────────────────────────────────────


def test_a_progress_iteration_does_not_count_toward_the_cap():
    from familiar_agent.loop.arbiter import Decision

    ip = _ip("発話", "search_deferred")

    async def decide(**_kw):
        return Decision(branch="full", effort="low", text="もう少し待ってね")

    ip._decide = decide
    ip._say_filler = _noop

    async def go():
        for _ in range(6):
            ip._drained_completions.append(Trigger(kind="進捗", query="q"))
            await ip._iterate()
        return ip._req.iterations, ip._req.iterations_capped

    assert asyncio.run(go()) == (0, False)


def test_a_real_result_wins_over_a_progress_notice():
    ip = _ip("発話", "search_deferred")

    async def go():
        ip._drained_completions.append(Trigger(kind="進捗", query="q"))
        ip._drained_completions.append(Trigger(kind="完了", query="q", result="晴れ", index=1))
        await ip._intake()
        return ip._slow_notice_received

    assert asyncio.run(go()) is False


async def _noop(*_a, **_kw):
    return None


# ── 待たせているあいだは窓を閉じない（2026-10-07 本人の決定・窓 10 秒）─────────────
#
# 窓を 30 秒から 10 秒に縮めた。つなぎは 5 秒・その後 20 秒ごとなので、10 秒の窓はその合間に切れ、時間のかかる問いの
# 答えが独り言になる。待たせているあいだは見張りが 1 秒ごと（試験では設定の短い値）に起き、窓を延ばす（本人「イで」）。


def _keeps(kind: str, *, run_for: float = 0.08):
    import time

    ip = _ip(kind, "search_deferred")
    w = ip._wake_window()
    w.open(time.monotonic() - 9.95)  # あと 0.05 秒で閉じる窓

    async def go():
        ip._ensure_wait_watch()
        await asyncio.sleep(run_for)
        ip._stop_wait_watch()

    asyncio.run(go())
    return w, time.monotonic()


def test_the_window_stays_open_while_a_conversation_waits():
    w, now = _keeps("発話")
    assert w.is_open(now)
    assert w.until - now > 9.0  # いまから 10 秒近く残っている（最後に延ばしたのは直前）


def test_the_window_closes_ten_seconds_after_the_wait_ends():
    w, now = _keeps("発話")
    assert not w.is_open(now + 10.5)


def test_affect_does_not_keep_the_window():
    w, now = _keeps("情動")
    assert not w.is_open(now)
