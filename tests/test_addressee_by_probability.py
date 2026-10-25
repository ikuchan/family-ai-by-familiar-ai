"""宛先の判定は、ロボット宛ての確率で決める（出-db 段 C・2026-10-25・本人の決定 3-a・3-b・3-d・`設計方針_判定の段` §2.2.3）。

10/23〜10/24、テレビの声や崩れた書き起こしが窓の中で「パジュ宛て」と読まれ（倒し先が「迷ったら受ける」だった）、聞き返しが
続いた。選択肢に「どこにも向いていない（テレビ・動画の声、独り言、意味の通らない書き起こし）」を足し、ロボット宛ての確率が
0.3 未満なら黙る。Jev が答えないときだけ、いまどおり受ける（返事を書く LLM が後で黙れる）。問いの文にパジュの名前は入れず、
ME.md の呼び名と声の似かたを材料に入れる。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges


def _client(answer):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=answer)
    return c


def _robot(p: float) -> JevAnswer:
    rest = (1.0 - p) / 3
    probs = {"robot": p, "family": rest, "nowhere": rest, "unknown": rest}
    top = max(probs, key=lambda k: probs[k])
    return JevAnswer(
        ok=True,
        answers={"to_whom": {"choice": top, "confidence": probs[top], "probabilities": probs}},
    )


def _judge(c, **kw):
    args = dict(
        text="ごはんできたよー",
        recent="- 相手：パジュ、今日の予定は？",
        present="パパ・ママ",
        names=["パジュ", "はじゅ"],
        voice="パパ 0.62・ママ 0.31",
    )
    args.update(kw)
    return asyncio.run(jev_judges.judge_addressee(c, **args))


@pytest.mark.parametrize(
    ("p", "want"), [(0.29, jev_judges.NOT_TO_PAJU), (0.30, jev_judges.TO_PAJU)]
)
def test_the_robot_probability_decides(p, want):
    assert _judge(_client(_robot(p))) == want


def test_a_low_probability_robot_choice_is_still_heard():
    """**反証の側**：確信度ではなく確率で見る。ロボット宛てが 1 番でも 0.3 以上で、確信度が低くても受ける。"""
    assert _judge(_client(_robot(0.4))) == jev_judges.TO_PAJU


def test_when_jev_does_not_answer_it_is_heard():
    assert _judge(_client(JevAnswer(ok=False, error="時間切れ"))) == jev_judges.TO_PAJU
    down = MagicMock()
    down.available = False
    assert _judge(down) == jev_judges.TO_PAJU


def test_the_choices_and_the_state():
    c = _client(_robot(0.9))
    _judge(c)
    state, questions = c.ask.await_args.args
    q = questions["to_whom"]
    assert set(q["criteria"]) == {"robot", "family", "nowhere", "unknown"}
    assert "テレビ" in q["criteria"]["nowhere"]
    assert "遊び（しりとりなど）の続きの一語" in q["criteria"]["robot"]
    assert "パジュ" not in q["instructions"] and "パジュ" not in "".join(q["criteria"].values())
    assert "[ロボットの呼び名]\nパジュ、はじゅ" in state
    assert "ごはんできたよー" in state and "今日の予定" in state and "ママ" in state
    assert "パパ 0.62" in state


def test_unknown_voice_is_written_as_unknown():
    c = _client(_robot(0.9))
    _judge(c, voice="")
    state, _ = c.ask.await_args.args
    assert "[声の似かた（家族の声との近さ）]\n（分からない）" in state


def test_the_probability_is_logged_without_the_words(caplog):
    with caplog.at_level(logging.INFO, logger="familiar_agent.core.jev_judges"):
        _judge(_client(_robot(0.25)), text="きょうも退屈してたんだ、遊んでやれ")
    line = next(r.getMessage() for r in caplog.records if "宛先" in r.getMessage())
    assert "0.25" in line and "退屈" not in line


def test_the_line_is_three_tenths():
    assert jev_judges.ADDRESSEE_ROBOT_MIN == 0.3


def test_the_loop_passes_the_names_and_the_voice_by_call_name():
    """呼び名は ME.md（`agent_names`）から、声は人物 id ではなく呼びかけ名で、近い順に渡す。"""
    from familiar_agent.loop.event_loop import InformationProcessing, Trigger
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.agent_names = ["パジュ", "パチュ"]
    a._jev = _client(_robot(0.9))
    ip = InformationProcessing(a)
    ip._voice_scores = lambda v: {"p1": 0.31, "p2": 0.62, "p3": 0.1}
    ip._call_name_for = lambda pid: {"p1": "ママ", "p2": "パパ"}.get(pid)
    t = Trigger(kind="会話入力", query="はい", source="voice", voice=object())
    assert asyncio.run(ip._between_family(t)) is False
    state, _ = a._jev.ask.await_args.args
    assert "[ロボットの呼び名]\nパジュ、パチュ" in state
    assert "パパ 0.62・ママ 0.31" in state and "p3" not in state


def test_a_voice_that_cannot_be_read_is_unknown():
    """**反証の側**：点数を読めなければ「分からない」と書く（止めない）。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))

    def boom(v):
        raise RuntimeError("no store")

    ip._voice_scores = boom
    assert ip._voice_line(object()) == ""
    assert ip._voice_line(None) == ""


def test_an_answer_without_probabilities_is_heard():
    """**反証の側**：答えに確率も選んだものも無ければ、答えなしと同じに受ける（確率 0 として黙らせない）。"""
    assert _judge(_client(JevAnswer(ok=True, answers={}))) == jev_judges.TO_PAJU


def test_the_same_call_name_is_listed_once():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._voice_scores = lambda v: {"p1": 0.4, "p2": 0.62}
    ip._call_name_for = lambda pid: "パパ"
    assert ip._voice_line(object()) == "パパ 0.62"


def test_a_failing_name_lookup_is_unknown():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._voice_scores = lambda v: {"p1": 0.4}

    def boom(pid):
        raise RuntimeError("db")

    ip._call_name_for = boom
    assert ip._voice_line(object()) == ""
