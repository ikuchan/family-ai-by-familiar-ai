"""発話前の検査でも、パジュ自身を呼び名で書く（記-x の続き・2026-11-03・本人の決定）。

W はパジュ自身を ME.md の呼び名の先頭（`me_name`）で書く（記-x）。検査の state も Jev が同じ反復で読むので、
返事の見出し（「家のロボット（…）がいま言おうとしている返事」）と、情動のきっかけの見出し（「…の中で起きたこと」）を
同じ名前にする。呼び名が無ければ「わたし」（本人の決定ア・`me_name` と同じ）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent
from tests.test_verdicts_come_from_jev_only import _c, _client
from tests.test_workspace_is_the_core import _rec

URGE = "[内的な促し:SAFETY] 確かめたい気持ちが湧いている。見回る。"


def _ip(names):
    a = _agent(stream_returns=[])
    a.config.agent_names = names
    a.config.speech_check = True
    a.config.jev_confidence_min = 0.6
    a._jev = _client({"broken": _c("ok")})
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.request_text = URGE
    ip._req.trigger_kind = "情動"
    return ip, a


def _state(ip, a):
    async def scenario():
        await ip._speech_check_violation("見回ったよ", "", [_rec("req-1", URGE, fit=0.9)])
        await ip.close()

    asyncio.run(scenario())
    state, _ = a._jev.ask.await_args.args
    return state


def test_the_cause_is_written_with_the_call_name():
    ip, _a = _ip(["ポポ", "ポポちゃん"])
    facts = ip._checker_facts([_rec("req-1", URGE, fit=0.9)])
    assert "この求めのきっかけ（ポポの中で起きたこと・根拠ではない）：" in facts
    assert "パジュの中で" not in facts


def test_the_reply_heading_is_written_with_the_call_name():
    ip, a = _ip(["ポポ"])
    state = _state(ip, a)
    assert "[家のロボット（ポポ）がいま言おうとしている返事]" in state
    assert "ポポの中で起きたこと" in state
    assert "パジュ" not in state


def test_without_names_it_is_watashi():
    """**反証の側**：呼び名が無ければ「わたし」（`me_name` と同じ・決定ア）。"""
    ip, a = _ip([])
    state = _state(ip, a)
    assert "[家のロボット（わたし）がいま言おうとしている返事]" in state
    assert "わたしの中で起きたこと" in state


def test_a_renamed_agent_is_read_at_the_next_check():
    """**反証の側**：ME.md を読み直して呼び名が変わったら、次の検査から新しい名前（毎回設定から読む）。"""
    ip, a = _ip(["ポポ"])
    a.config.agent_names = ["ピピ"]
    state = _state(ip, a)
    assert "[家のロボット（ピピ）がいま言おうとしている返事]" in state
    assert "ポポ" not in state
