"""調停で Jev に聞いたら、渡したそのままと答え・道・意味・最終の動作を残す（出-ay 段 3・段 5d・本人の決定）。

正解（`JEV_正解.md`）の時刻から、そのとき Jev に渡した文（W を含む）を引き、前提を足して投げ直せるようにする。
段 5d（2026-10-10）で、起点ごとの新しい道（発話・完了・情動）が書くようにした。**Jev に聞かなかった回は残さない**（本人の
決定イ：機器・選択肢が 1 つの完了・seeking）。記録に失敗しても調停は止めない。会話の中身はログに出さず、DB の中だけに置く。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import writer_says


def _c(pick, conf=0.9):
    return {"choice": pick, "confidence": conf}


def _jev(*answers):
    queue = [JevAnswer(ok=True, answers=a) for a in answers]
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(side_effect=lambda s, q: queue.pop(0) if len(queue) > 1 else queue[0])
    return c


def _decide(jev, **kw):
    writer = writer_says({"text": "もう一度お願いします", "query": "明日の天気"})
    arb = Arbiter(jev=jev, writer=writer)
    base = dict(utterance="パジュ、明日の天気は？", workspace_ctx="[直近のやりとり]", origin="発話")

    async def go():
        got = await arb.decide(ArbiterInput(**{**base, **kw}))
        await arbiter_records.flush()
        return got

    return asyncio.run(go())


def _rows():
    now = datetime.now(timezone.utc)
    return arbiter_records.near(now - timedelta(seconds=30), seconds=120)


def test_an_utterance_is_recorded_with_what_jev_saw():
    jev = _jev({"meaning": _c("research"), "action_research": _c("search_deferred")})
    _decide(jev)
    (row,) = _rows()
    state, questions = jev.ask.await_args.args
    assert row["state"] == state  # 渡したそのまま（W を含む）
    assert set(row["questions"]) == set(questions)
    assert row["utterance"] == "パジュ、明日の天気は？" and row["origin"] == "発話"
    assert (row["path"], row["meaning"], row["final"]) == ("発話", "research", "search_deferred")
    assert row["answer"]["meaning"]["choice"] == "research"


def test_an_unsure_utterance_records_the_second_answer_and_why():
    jev = _jev({"meaning": _c("research", 0.4)}, {"action": _c("ask_back", 0.3)})
    _decide(jev)
    (row,) = _rows()
    assert row["final"] == "ask_back"
    assert row["answer"]["unsure"]["action"]["choice"] == "ask_back"
    assert "しきい値未満" in row["outcome"]


def test_a_completion_asked_to_jev_is_recorded_as_a_completion():
    jev = _jev({"action": _c("reply_light", 0.4)})
    _decide(jev, returned=(("search_deferred", False, "明日は晴れ"),))
    (row,) = _rows()
    assert (row["path"], row["meaning"], row["final"]) == ("完了", "", "reply_light")


def test_an_affect_asked_to_jev_is_recorded():
    jev = _jev({"action": _c("look", 0.3)})
    _decide(jev, origin="情動", fired_axis="safety", utterance="[内的な促し:SAFETY] 見たい")
    (row,) = _rows()
    assert (row["path"], row["final"]) == ("情動", "look")


def test_what_was_decided_without_jev_is_not_recorded():
    jev = _jev({"action": _c("look")})
    _decide(jev, origin="情動", fired_axis="seeking", utterance="[内的な促し:SEEKING] 知りたい")
    _decide(jev, returned=(("play_music", False, "かけ始めた"),))
    _decide(jev, origin="機器", utterance="タイマーが鳴った")
    jev.ask.assert_not_awaited()
    assert _rows() == []


def test_a_failed_write_does_not_stop_the_arbiter(monkeypatch):
    def broken(**kw):
        raise RuntimeError("DB が無い")

    monkeypatch.setattr(arbiter_records, "_insert", broken)
    jev = _jev({"meaning": _c("answerable"), "action_answerable": _c("reply_full")})
    decision = _decide(jev)
    assert decision.branch == "full"
