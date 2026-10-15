"""say の無い文は、Jev が返事と読んだら主LLM へ 1 回だけ「say で言って」と差し戻す（出-ck・2026-10-15・本人の決定チ）。

10/14 23:03 に、主LLM が調べた結果の答えを say を呼ばずに地の文で返し、声にならなかった（会話で 95 回中 1 回）。
文そのものを機械が声にはしない——独り言として書いた文を外に出すことになる（本人）。Jev が「いまの相手への返事」と
読んだときだけ、主LLM 自身に say で言い直させる。言い直しは発話前の検査を通す。言い直しも文だけなら沈黙で閉じる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends import ToolCall
from familiar_agent.backends.jev import JevAnswer
from familiar_agent.backends.types import TurnResult
from familiar_agent.core import jev_judges
from familiar_agent.loop.event_loop import Decision
from tests.test_send_back_is_dispatched import _ip

_BARE = "プレミアリーグの 1 位はアーセナルだよ"


def _client(pick: "str | None", conf: float = 0.9, *, available: bool = True):
    c = MagicMock()
    c.available = available
    c.ask = AsyncMock(
        return_value=JevAnswer(ok=True, answers={"reply": {"choice": pick, "confidence": conf}})
    )
    return c


# --- 1. 判定 ---


def _judge(client) -> bool:
    return asyncio.run(
        jev_judges.judge_bare_reply(
            client, heard="プレミアリーグの1位は？", text=_BARE, min_conf=0.6
        )
    )


def test_a_reply_is_read_as_a_reply():
    c = _client("reply")
    assert _judge(c) is True
    state, questions = c.ask.await_args.args
    assert "プレミアリーグの1位は？" in state and _BARE in state
    assert set(questions["reply"]["criteria"]) == {"reply", "monologue"}


def test_monologue_low_confidence_and_no_jev_are_not_replies():
    assert _judge(_client("monologue")) is False
    assert _judge(_client("reply", conf=0.59)) is False
    assert _judge(_client("reply", available=False)) is False
    assert _judge(None) is False


# --- 2〜4. 出す反復 ---


def _bare(text: str = _BARE) -> TurnResult:
    return TurnResult(stop_reason="end_turn", text=text, tool_calls=[])


def _act(ip, result, *, kind="発話", axis="", retried=False, say_asked=False):
    ip._req.trigger_kind = kind
    ip._req.fired_axis = axis
    return asyncio.run(
        ip._act_on_decision(
            Decision(
                result=result,
                memories=[],
                w_id_map={},
                verdict_view="__self__",
                recent_frame="",
                system=("安定", "可変"),
                effort="low",
                capped=False,
                retried=retried,
                say_asked=say_asked,
            ),
            utterance="プレミアリーグの1位は？",
            gen=0,
        )
    )


def _with_jev(pick: str):
    ip, a = _ip()
    a._jev = _client(pick)
    a.config.jev_confidence_min = 0.6
    return ip, a


def test_a_bare_reply_is_sent_back_once_to_say_it():
    ip, a = _with_jev("reply")
    assert _act(ip, _bare()) == ""
    ip._dispatch_main_llm.assert_called_once()
    kw = ip._dispatch_main_llm.call_args.kwargs
    assert kw["say_asked"] is True and kw.get("retried", False) is False
    sent = kw["messages"][0]["content"]
    assert "say()" in sent and _BARE in sent
    ip._speak.assert_not_awaited()  # 文そのものは声にしない
    ip._finish.assert_not_awaited()  # 閉じない（言い直しを待つ）


def test_a_monologue_closes_in_silence_as_before():
    """**反証の側**：独り言と読んだら、いまどおり沈黙で閉じ、差し戻さない。"""
    ip, a = _with_jev("monologue")
    _act(ip, _bare())
    ip._dispatch_main_llm.assert_not_called()
    assert ip._finish.await_args.args[2] == "沈黙"


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": "情動", "axis": "safety"},  # どのみち声にしない軸
        {"say_asked": True},  # 差し戻しの返りも文だけ
        {"retried": True},  # 発話前の検査の言い直しの返り
    ],
)
def test_no_question_where_it_cannot_or_must_not_be_sent_back(kw):
    """**反証の側**：Jev に聞かず、沈黙で閉じる。元の文は声にしない。"""
    ip, a = _with_jev("reply")
    _act(ip, _bare(), **kw)
    a._jev.ask.assert_not_awaited()
    ip._dispatch_main_llm.assert_not_called()
    ip._speak.assert_not_awaited()
    assert ip._finish.await_args.args[2] == "沈黙"


def test_a_talking_affect_is_asked_too():
    ip, a = _with_jev("reply")
    _act(ip, _bare(), kind="情動", axis="bond")
    ip._dispatch_main_llm.assert_called_once()


def test_the_resaid_answer_goes_through_the_speech_check():
    ip, a = _with_jev("reply")
    said = TurnResult(
        stop_reason="tool_use", text="", tool_calls=[ToolCall("t", "say", {"text": _BARE})]
    )
    _act(ip, said, say_asked=True)
    ip._speech_check_violation.assert_awaited_once()
    ip._speak.assert_awaited_once()


def test_the_sent_back_call_offers_only_say():
    """差し戻した呼び出しで渡す道具は、言い直しと同じく say（とつなぎ）だけ。"""
    from familiar_agent.loop import event_loop

    src = __import__("inspect").getsource(event_loop.InformationProcessing._run_main_llm)
    assert "capped or retried or say_asked" in src


def test_the_sent_back_is_logged_without_the_text(caplog):
    import logging

    ip, a = _with_jev("reply")
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        _act(ip, _bare())
    lines = [r.getMessage() for r in caplog.records if "say を呼ばずに" in r.getMessage()]
    assert lines and "返事" in lines[0] and _BARE not in " ".join(lines)
