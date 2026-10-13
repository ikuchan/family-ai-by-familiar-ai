"""意図の台帳の記録（`store/intents`・出-cc・2026-10-13・本人の決定）。

投げたときに 1 行足し、終わったときに状態・失敗の印・結果の字数・終わった時刻を書く。書くのは別スレッドで、順番は
崩さない（終わりの更新が始まりの追加より先に走ると 0 行の更新になる）。書けなくても例外を外へ出さない。
"""

from __future__ import annotations

import asyncio
import os
import uuid

import psycopg2
import psycopg2.extras

from familiar_agent.store import intents

_DB_URL = os.environ["DATABASE_URL"]


def _row(intent_id: str) -> dict:
    with psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM intents WHERE id = %s", (intent_id,))
            got = cur.fetchone()
    return dict(got) if got else {}


def test_an_intent_is_opened_and_ended_in_order():
    iid = str(uuid.uuid4())

    async def go():
        intents.opened(
            iid, request_id="req-1", origin="発話", action="set_timer", query="タイマーを掛ける"
        )
        intents.ended(iid, state=intents.RETURNED, failed=False, result_chars=12)
        await intents.flush()

    asyncio.run(go())
    row = _row(iid)
    assert (row["request_id"], row["origin"], row["action"]) == ("req-1", "発話", "set_timer")
    assert (row["state"], row["failed"], row["result_chars"]) == (intents.RETURNED, False, 12)
    assert row["ended_at"] is not None and row["ended_at"] >= row["opened_at"]


def test_an_open_intent_has_no_end():
    iid = str(uuid.uuid4())

    async def go():
        intents.opened(iid, request_id="req-2", origin="情動", action="look", query="出入口")
        await intents.flush()

    asyncio.run(go())
    row = _row(iid)
    assert row["state"] == intents.OPEN and row["ended_at"] is None


def test_a_broken_db_does_not_raise(monkeypatch, caplog):
    """**反証の側**：書けなくても例外を外へ出さず、警告 1 行だけ。"""

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(intents, "_insert", boom)
    intents.opened(str(uuid.uuid4()), request_id="r", origin="発話", action="recall", query="x")
    assert "意図の記録を残せなかった" in caplog.text


def test_the_states_are_named():
    assert (intents.OPEN, intents.RETURNED, intents.ABORTED, intents.CLOSED, intents.LATE) == (
        "開いている",
        "結果が届いた",
        "打ち切った",
        "求めと一緒に閉じた",
        "閉じた後に届いた",
    )
