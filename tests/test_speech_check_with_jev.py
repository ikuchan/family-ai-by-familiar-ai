"""発話前の検査を Jev が決める（出-au 段 5-3・2026-09-27・`設計方針_判定の段` §2.2.3）。

返事が主LLM のシステム文の決まり（`(rules …)` の `constraint`）を破っていないかを、Jev の Choice で聞く。選択肢は
決まり（id → 説明）と「破っていない」。**決まりの一覧はシステム文から機械で読む**（2 か所に書かない）。

- 破っていれば、差し戻しの文は「決まり <id>：<説明>」（本人の決定 ア。どう反しているかの一文は書かない）。
- 確信度 0.6 未満・失敗なら倒し先の「破っていない」（迷っただけで主LLM を呼び直さない）。
- 送る文は、機械が集めた事実・直近のやりとり・返事。
- 軽量LLM の口（`Evaluator.check_speech`）は外した。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.prompt import rule_list

from tests.test_event_loop import _agent


def _client(answer=None):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=answer)
    return c


def _answer(pick, conf=0.8):
    return JevAnswer(ok=True, answers={"broken": {"choice": pick, "confidence": conf}})


def test_the_rules_are_read_from_the_system_prompt():
    rules = rule_list()
    assert "voice-only-from-say" in rules and "no-fake-perception" in rules
    assert rules["no-fake-perception"].startswith("見たと言えるのは")
    assert len(rules) >= 15


def _judge(client, min_conf=0.6):
    return asyncio.run(
        jev_judges.judge_speech(
            client,
            response="そこに本があるね",
            recent="- 相手：何が見える？",
            facts="見たか：いいえ",
            rules={"no-fake-perception": "見たと言えるのは写真に写っていることだけ"},
            min_conf=min_conf,
        )
    )


def test_a_broken_rule_becomes_the_send_back_text():
    got = _judge(_client(_answer("no-fake-perception")))
    assert got == "決まり no-fake-perception：見たと言えるのは写真に写っていることだけ"


def test_ok_low_confidence_and_failure_mean_nothing_is_broken():
    assert _judge(_client(_answer("ok"))) is None
    assert _judge(_client(_answer("no-fake-perception", 0.5))) is None
    assert _judge(_client(JevAnswer(ok=False, error="時間切れ"))) is None
    assert _judge(None) is None


def test_the_state_carries_facts_recent_and_the_response():
    c = _client(_answer("ok"))
    _judge(c)
    state, questions = c.ask.await_args.args
    assert "見たか：いいえ" in state and "何が見える？" in state and "そこに本があるね" in state
    assert set(questions["broken"]["criteria"]) == {"no-fake-perception", "ok"}


def test_the_loop_asks_jev_with_the_facts():
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    a.config.jev_confidence_min = 0.6
    a._jev = _client(_answer("no-fake-perception"))
    ip = InformationProcessing(a)

    async def go():
        got = await ip._speech_check_violation("はい", "", [])
        await ip.close()
        return got

    got = asyncio.run(go())
    assert got is not None and got.startswith("決まり no-fake-perception：")
    # 照らすのは文と事実だけで判断できる決まりだけ（`CHECKER_RULE_IDS`・出-n）
    _state, questions = a._jev.ask.await_args.args
    assert "voice-only-from-say" not in questions["broken"]["criteria"]


def test_the_light_llm_check_is_gone():
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop import evaluator

    assert not hasattr(evaluator.Evaluator, "check_speech")
    assert not hasattr(evaluator, "_SPEECH_CHECK_PROMPT")
    assert not hasattr(EmbodiedAgent, "_check_speech")
