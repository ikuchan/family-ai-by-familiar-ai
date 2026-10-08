"""調停のたびに、Jev に渡したそのままと答えを残す（出-ay 段 3・2026-10-08・本人の決定）。

正解（`JEV_正解.md`）の時刻から、そのとき Jev に渡した文（W を含む）を引き、前提を足して投げ直せるようにする。
記録に失敗しても調停は止めない。会話の中身はログに出さず、DB の中だけに置く。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter
from familiar_agent.store import arbiter_records

from tests.test_arbiter_judges_with_jev import _c, _inp, _p


def _jev(answers):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    return c


def _decide(jev, inp=None):
    arb = Arbiter(jev=jev, writer=MagicMock(), min_conf=0.6)

    async def go():
        got = await arb.decide(inp or _inp(utterance="パジュ、明日の天気は？"))
        await arbiter_records.flush()
        return got

    return asyncio.run(go())


def _rows():
    now = datetime.now(timezone.utc)
    return arbiter_records.near(now - timedelta(seconds=30), seconds=120)


def test_each_judgement_is_recorded_with_what_jev_saw():
    jev = _jev({"branch": _c("full", 0.9), "effort": _c("low"), "action": _c("recall", 0.3)})
    _decide(jev)
    (row,) = _rows()
    state, questions = jev.ask.await_args.args
    assert row["state"] == state  # 渡したそのまま（W を含む）
    assert set(row["questions"]) == set(questions)
    assert row["utterance"] == "パジュ、明日の天気は？" and row["origin"] == "発話"
    assert row["answer"]["branch"]["choice"] == "full"
    assert row["answer"]["action"]["confidence"] == 0.3
    assert row["outcome"].startswith("full")


def test_a_fallback_records_why():
    jev = _jev({"branch": _p("action", 0.33, {"action": 0.55, "full": 0.31})})
    _decide(jev)
    (row,) = _rows()
    assert row["outcome"].startswith("倒れ：分岐の確信度 0.33")


def test_a_failed_write_does_not_stop_the_arbiter(monkeypatch):
    def broken(**kw):
        raise RuntimeError("DB が無い")

    monkeypatch.setattr(arbiter_records, "_insert", broken)
    jev = _jev({"branch": _c("full", 0.9), "effort": _c("low")})
    decision = _decide(jev)
    assert decision.branch == "full"
