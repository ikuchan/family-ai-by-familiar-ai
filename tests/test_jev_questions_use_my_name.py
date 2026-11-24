"""記-z：Jev に送る問いと材料は、パジュ自身を呼び名で書く（2026-11-21）。

記-x とその続きで、W と発話前の検査は `me_name`（呼び名の先頭・無ければ「わたし」）で書くようになった。Jev のほかの判定は
「パジュ」と決め打ちで、呼び名を変えると、W と同じ反復で読まれる問いで名前がずれた。名前は呼び手が W と同じ読み元
（`me_name(config.agent_names)`）から `me` で渡す。渡さなければ「わたし」。
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace


from familiar_agent.core import jev_judges as jj


class _Spy:
    available = True

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def ask(self, state, questions):
        self.sent.append(str(state) + "\n" + str(questions))
        return SimpleNamespace(ok=False, answers={}, model="t")


def _all_judges(c):
    return [
        jj.judge_rethink(c, me="ルル", question="q", draft="d", added=["a"], min_conf=0.6),
        jj.judge_confirm_answer(c, me="ルル", question="q", text="はい", min_conf=0.6),
        jj.judge_bare_reply(c, me="ルル", heard="h", text="t", min_conf=0.6),
        jj.judge_verdicts(
            c,
            me="ルル",
            utterance="u",
            reply="r",
            workspace_ctx="w",
            ids=["1"],
            min_conf=0.6,
            rows={"1": "x"},
        ),
        jj.judge_emotion(
            c,
            me="ルル",
            text="t",
            arousal=0.5,
            a_gate=0.5,
        ),
        jj.judge_music_pick(
            c,
            me="ルル",
            candidates=[{"title": "a", "artist": "b", "reason": "r", "uri": "u"}],
            family_now="f",
            artists="a",
            previous="",
        ),
        jj.judge_season_reads(
            c,
            me="ルル",
            today=date(2026, 11, 21),
            place="p",
            news=[SimpleNamespace(title="n")],
            columns=[SimpleNamespace(title="c")],
        ),
    ]


def test_every_question_uses_the_name_it_is_given():
    c = _Spy()

    async def go():
        for coro in _all_judges(c):
            try:
                await coro
            except Exception:  # noqa: BLE001  返りを読む側の都合で落ちても、送った文は見られる
                pass

    asyncio.run(go())
    assert len(c.sent) == 7
    for sent in c.sent:
        assert "パジュ" not in sent, sent[:200]
    assert sum("ルル" in s for s in c.sent) == 7


def test_without_a_name_it_is_watashi():
    c = _Spy()
    asyncio.run(jj.judge_confirm_answer(c, question="q", text="はい", min_conf=0.6))
    assert "家のロボット（わたし）" in c.sent[0]


def test_the_arbiter_and_evaluator_pass_the_name_from_config():
    """本番の組み立ては、W と同じ `config.agent_names` から名前を渡す（読み元が 1 つ）。"""
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.agent_names = ["ルル", "ルルちゃん"]
    ip = InformationProcessing(a)
    assert ip._arbiter()._me() == "ルル"
    assert EmbodiedAgent._me_name(a) == "ルル"
    a._me_name = lambda: EmbodiedAgent._me_name(a)
    a.__dict__.pop("_evaluator_obj", None)
    assert EmbodiedAgent._evaluator.fget(a)._me() == "ルル"  # 評価も同じ名前を渡す
