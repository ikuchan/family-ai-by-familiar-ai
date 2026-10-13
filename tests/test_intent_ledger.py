"""意図の台帳（`loop/intent_ledger`・出-cc・2026-10-13・本人の決定）。

意図の器（`Lookup`）は作ったときに uuid を持ち、パジュ全体（装置の寿命）の台帳に載る。完了は uuid で器を引き、結果が届いた・
打ち切った・求めを閉じたで台帳から外れる。始まりと終わりは DB（`store/intents`）へ記録として渡す。
"""

from __future__ import annotations

import uuid

from familiar_agent.loop.intent_ledger import IntentLedger
from familiar_agent.loop.request import Lookup
from familiar_agent.store import intents


class _Store:
    def __init__(self):
        self.calls: list = []

    def opened(self, intent_id, **kw):
        self.calls.append(("opened", intent_id, kw))

    def ended(self, intent_id, **kw):
        self.calls.append(("ended", intent_id, kw))


def _lk(query="天気", action="search_deferred", index=1):
    return Lookup(index=index, action=action, query=query, generation=0)


def test_a_lookup_has_its_own_uuid():
    a, b = _lk(), _lk()
    assert uuid.UUID(a.intent_id) and a.intent_id != b.intent_id


def test_open_get_close():
    store = _Store()
    ledger = IntentLedger(store=store)
    lk = _lk()
    ledger.open(lk, request_id="req-1", origin="発話")
    assert ledger.get(lk.intent_id) is lk
    assert store.calls[0] == (
        "opened",
        lk.intent_id,
        {"request_id": "req-1", "origin": "発話", "action": "search_deferred", "query": "天気"},
    )
    ledger.close(lk.intent_id, state=intents.RETURNED, failed=False, result="晴れ")
    assert ledger.get(lk.intent_id) is None
    assert store.calls[1] == (
        "ended",
        lk.intent_id,
        {"state": intents.RETURNED, "failed": False, "result_chars": 2},
    )


def test_an_unknown_id_is_not_found():
    """**反証の側**：台帳に無い uuid・None は引けない。"""
    ledger = IntentLedger(store=_Store())
    assert ledger.get(str(uuid.uuid4())) is None and ledger.get(None) is None


def test_a_request_closes_its_open_intents():
    store = _Store()
    ledger = IntentLedger(store=store)
    open_, done = _lk(index=1), _lk(query="音楽", index=2)
    for lk in (open_, done):
        ledger.open(lk, request_id="req-1", origin="発話")
    ledger.close(done.intent_id, state=intents.RETURNED, failed=False, result="かけた")
    ledger.close_request([open_, done], state=intents.ABORTED)
    assert ledger.get(open_.intent_id) is None
    ended = [c for c in store.calls if c[0] == "ended"]
    assert [c[1] for c in ended] == [done.intent_id, open_.intent_id]  # 閉じたものは二度閉じない
    assert ended[-1][2]["state"] == intents.ABORTED


def test_a_late_completion_is_recorded():
    store = _Store()
    ledger = IntentLedger(store=store)
    lk = _lk()
    ledger.open(lk, request_id="req-1", origin="情動")
    ledger.close_request([lk], state=intents.CLOSED)
    ledger.late(lk.intent_id)
    assert store.calls[-1] == (
        "ended",
        lk.intent_id,
        {"state": intents.LATE, "failed": False, "result_chars": None},
    )


def test_a_closed_lookup_can_be_reopened():
    store = _Store()
    ledger = IntentLedger(store=store)
    lk = _lk()
    ledger.open(lk, request_id="req-1", origin="発話")
    ledger.close(lk.intent_id, state=intents.RETURNED, result="x")
    ledger.reopen(lk)
    assert ledger.get(lk.intent_id) is lk
    assert [c[0] for c in store.calls] == ["opened", "ended"]  # 記録は書き直さない
