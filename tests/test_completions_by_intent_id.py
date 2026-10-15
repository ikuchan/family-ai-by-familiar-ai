"""完了は意図の uuid で器を引き、台帳に無い完了は捨てる（出-cc・2026-10-13・本人の決定）。

10/13 20:00:33 に、閉じた求めの `look` の完了が「対応する意図が無い完了を捨てる」と出ながら取込に数えられ、想起と調停が
回ってタイマー中に話した。語で引いていたので、前の求めと同じ語の結果が新しい器に当たることもあった。
"""

from __future__ import annotations

import asyncio
import inspect
import re

from familiar_agent.loop import event_loop
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.intent_ledger import IntentLedger
from familiar_agent.store import intents

from tests.test_confirm_is_exclusive import _drive_once
from tests.test_event_loop import _agent
from tests.test_intent_ledger import _Store


def _ip():
    ip = InformationProcessing(_agent(stream_returns=[]))
    store = _Store()
    ip._ledger = IntentLedger(store=store)
    ip._req.request_id = "req-1"
    ip._req.trigger_kind = "発話"
    return ip, store


def test_a_lookup_is_opened_in_the_ledger():
    ip, store = _ip()
    lk = ip._open_lookup("search_deferred", "天気")
    assert lk in ip._req.lookups and ip._ledger.get(lk.intent_id) is lk
    assert store.calls[0][0] == "opened" and store.calls[0][2]["request_id"] == "req-1"


def test_the_same_words_from_an_old_request_do_not_fill_the_new_lookup():
    ip, store = _ip()
    lk = ip._open_lookup("search_deferred", "天気")
    ip._drained_completions.append(
        Trigger(kind="完了", query="天気", result="昨日の", intent_id="old-id")
    )
    assert ip._drop_orphan_completions() is True  # 意図の無い完了だけ → 反復を起こさない
    assert lk.result is None
    assert store.calls[-1] == (
        "ended",
        "old-id",
        {"state": intents.LATE, "failed": False, "result_chars": None},
    )


def test_the_right_id_fills_the_lookup():
    """**反証の側**：正しい uuid なら器を埋め、台帳から外す。"""
    ip, store = _ip()
    lk = ip._open_lookup("search_deferred", "天気")
    ip._drained_completions.append(
        Trigger(kind="完了", query="天気", result="晴れ", intent_id=lk.intent_id)
    )
    assert ip._drop_orphan_completions() is False
    took, _ = asyncio.run(ip._intake())
    assert took == 1 and lk.result == "晴れ"
    assert ip._ledger.get(lk.intent_id) is None
    assert store.calls[-1][2]["state"] == intents.RETURNED


def test_an_orphan_does_not_iterate_outside_a_confirmation():
    ip, _ = _ip()
    ip._agent._pending_confirm = None
    ip._drained_completions.append(
        Trigger(kind="完了", query="出入口を見に行く", result="見た", intent_id="gone")
    )
    assert "to" not in _drive_once(ip, None)


def test_a_completion_without_an_id_still_finds_its_lookup_by_words():
    """uuid を持たない完了（試験の組み立てなど）は語で引く。器があれば埋める。"""
    ip, _ = _ip()
    lk = ip._open_lookup("recall", "去年の夏")
    ip._drained_completions.append(Trigger(kind="完了", query="去年の夏", result="海"))
    assert ip._drop_orphan_completions() is False
    asyncio.run(ip._intake())
    assert lk.result == "海"


def test_push_completion_drops_an_unknown_id():
    ip, store = _ip()
    lk = ip._open_lookup("search_deferred", "天気")
    ip.push_completion("天気", "晴れ", intent_id="nope")
    assert ip._triggers.empty()
    assert store.calls[-1][1] == "nope" and store.calls[-1][2]["state"] == intents.LATE
    ip.push_completion("天気", "晴れ", intent_id=lk.intent_id)
    assert ip._triggers.get_nowait().intent_id == lk.intent_id


def test_every_completion_carries_its_intent_id():
    """本体で完了・決定・進捗を積むところは、すべて uuid を渡す（語で引く道は試験のためだけに残る）。"""
    src = inspect.getsource(event_loop)
    calls = re.findall(
        r"Trigger\((?:[^()]|\([^()]*\))*?kind=\"(?:完了|決定|進捗)\"(?:[^()]|\([^()]*\))*?\)", src
    )
    assert calls, "完了を積むところが見つからない"
    missing = [c[:80] for c in calls if "intent_id=" not in c]
    assert missing == []


def test_every_clear_closes_the_ledger_first():
    src = inspect.getsource(event_loop)
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "self._req.lookups.clear()" in line:
            window = "\n".join(lines[max(0, i - 4) : i])
            assert "self._ledger.close_request(" in window, f"台帳を閉じずに器を消している：{i}"


def test_asking_the_same_words_again_is_not_dropped():
    """同じ求めで同じ語をもう一度頼むと、前の器で前の結果を返す。前の意図は閉じているので、台帳に戻して引けるようにする。"""
    ip, _ = _ip()
    lk = ip._open_lookup("recall", "q")
    ip._drained_completions.append(
        Trigger(kind="完了", query="q", result="前の", intent_id=lk.intent_id)
    )
    asyncio.run(ip._intake())
    assert ip._ledger.get(lk.intent_id) is None  # 閉じた
    ip._dispatch_lookup("recall", {"query": "q"}, "q", None)  # すでに調べた語
    again = ip._triggers.get_nowait()
    ip._drained_completions.append(again)
    assert ip._drop_orphan_completions() is False
