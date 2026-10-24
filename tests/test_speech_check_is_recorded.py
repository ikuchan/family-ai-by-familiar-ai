"""発話前の検査で Jev に聞いた回を、調停の記録（`arbiter_records`）に残す（知-bn・2026-10-24・本人の決定ア-1）。

調べものの返事は 10/22〜10/23 の 3 回とも `no-invented-knowledge` で差し戻され、主LLM の書き直しで 3.5〜10 秒待たせた。
差し戻した返事はどこにも残らず、検査が正しいのか厳しすぎるのかを決められなかった。

- Jev に渡した文（事実・直近・言おうとした返事）・問い・答え・違反の決まりを、道「発話前の検査」として 1 行書く。
- 通った回も残す（比べないと厳しすぎるかを決められない）。Jev に聞かなかった回は残さない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.store import arbiter_records

from tests.test_event_loop import _agent

RULES = {"no-fake-perception": "見たと言えるのは写真に写っていることだけ"}


def _client(answer=None):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=answer)
    return c


def _answer(pick, conf=0.8):
    return JevAnswer(ok=True, answers={"broken": {"choice": pick, "confidence": conf}})


def _judge(client, asked):
    return asyncio.run(
        jev_judges.judge_speech_and_verdicts(
            client,
            response="そこに本があるね",
            recent="- 相手：何が見える？",
            facts="見たか：いいえ",
            rules=RULES,
            rows={},
            min_conf=0.6,
            on_asked=lambda state, questions, answer: asked.append((state, questions, answer)),
        )
    )


# ── judge は聞いた中身を呼び手へ渡す ─────────────────────────────


def test_the_check_hands_what_it_asked():
    asked: list = []
    _judge(_client(_answer("no-fake-perception")), asked)
    [(state, questions, answer)] = asked
    assert "そこに本があるね" in state and "見たか：いいえ" in state
    assert "broken" in questions
    assert answer.answers["broken"]["choice"] == "no-fake-perception"


def test_nothing_is_handed_when_jev_was_not_asked():
    """**反証の側**：Jev が使えない回は聞いていないので渡さない。"""
    asked: list = []
    down = MagicMock()
    down.available = False
    _judge(down, asked)
    _judge(None, asked)
    assert asked == []


# ── ループは記録に残す ─────────────────────────────────────────


def _check(monkeypatch, answer, *, speech_check=True):
    kept: list = []
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: kept.append(kw))
    a = _agent(stream_returns=[])
    a.config.speech_check = speech_check
    a.config.jev_confidence_min = 0.6
    a._jev = _client(answer)
    ip = InformationProcessing(a)

    async def go():
        got = await ip._speech_check_violation("本があるね", "", [])
        await ip.close()
        return got

    asyncio.run(go())
    return kept


def test_a_sent_back_answer_is_recorded(monkeypatch):
    [row] = _check(monkeypatch, _answer("no-fake-perception"))
    assert row["path"] == "発話前の検査"
    assert row["outcome"].startswith("決まり no-fake-perception：")
    assert row["final"] == "差し戻した"
    assert "本があるね" in row["state"]
    assert "broken" in row["questions"]


def test_a_passed_answer_is_recorded_too(monkeypatch):
    """**反証の側**：通った回も残す（差し戻した回だけでは、厳しすぎるかを比べられない）。"""
    [row] = _check(monkeypatch, _answer("ok"))
    assert row["path"] == "発話前の検査"
    assert row["outcome"] == "破っていない"
    assert row["final"] == "話す"


def test_nothing_is_recorded_when_the_check_is_off(monkeypatch):
    assert _check(monkeypatch, _answer("no-fake-perception"), speech_check=False) == []
