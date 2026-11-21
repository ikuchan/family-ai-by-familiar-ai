"""情-u ウ：人に頼まれて黙っているあいだ、解く言い回しに当たらない名前つきの声は、Jev に「黙るのをやめてまた話してよいと
許しているか」を聞き、はいの確率が `silence_release_min`（0.7・仮）以上なら、その場で黙るのを終えて通す（ア・確かめずに
終える）。名前の無い声と、解く言い回しに当たる声は聞かない。Jev が答えないときは黙ったまま（いままでどおり控える）。

境目の試し（2026-11-21・`jev-1.14.0`・作り物の言葉 30 個）：許す 15 個は 0.23〜0.96（0.7 以上が 13 個。届かないのは
「お話ししよう」0.64 と「戻ってきて」0.23）、許さない 15 個は 0.02〜0.67（いちばん高いのは「しりとりしよう」）。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from familiar_agent.loop.event_loop import Trigger
from familiar_agent.silence_state import SilenceRequest
from tests.test_a_near_name_needs_a_meaning import _gate

pytestmark = pytest.mark.real_window


class _Jev:
    available = True

    def __init__(self, p: "float | None"):
        self._p = p
        self.calls: list = []

    async def ask(self, state, questions):
        self.calls.append((state, dict(questions)))
        if self._p is None:
            return SimpleNamespace(ok=False, answers={}, model="jev-test")
        return SimpleNamespace(ok=True, answers={"release": {"noul": self._p}}, model="jev-test")


def _silenced(jev):
    ip = _gate(jev, known_voice=True)
    ip._agent.config.silence_release_min = 0.7

    async def hush():
        await ip._paju().hush_with(SilenceRequest(person="", until=time.time() + 1800, reason="人"))

    asyncio.run(hush())
    return ip


def _say(ip, text):
    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(
            kind="会話入力", query=text, future=fut, source="voice", no_speech=0.01, logprob=-0.2
        )
        passed = not await ip._swallow_if_unheard(t)
        return passed, ip._paju().silence_request()

    return asyncio.run(go())


def _release_calls(jev):
    return [q for _s, q in jev.calls if "release" in q]


def test_a_named_permission_jev_says_yes_to_lifts_the_silence():
    jev = _Jev(0.9)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、もう怒ってないから声出して")
    assert passed is True and req is None
    assert len(_release_calls(jev)) == 1


def test_below_the_line_it_stays_silent():
    jev = _Jev(0.6)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、しりとりしよう")
    assert passed is False and req is not None


def test_no_answer_keeps_it_silent():
    jev = _Jev(None)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、もう怒ってないから声出して")
    assert passed is False and req is not None


def test_an_unnamed_voice_is_not_asked():
    jev = _Jev(0.99)
    ip = _silenced(jev)
    passed, req = _say(ip, "もう怒ってないから声出して")
    assert passed is False and req is not None
    assert _release_calls(jev) == []


def test_a_release_phrase_is_asked_too():
    """「話していいよ」の形も字だけでは解かず、Jev に聞く（2026-11-21 本人の決定 A）。"""
    jev = _Jev(0.9)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、話していいよ")
    assert passed is True and req is None
    assert len(_release_calls(jev)) == 1


def test_a_question_that_only_looks_like_a_release_stays_silent():
    """「パジュ、ママと話してもいい？」は字の形では解く言葉に当たるが、Jev が許しと読まなければ黙ったまま。"""
    jev = _Jev(0.1)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、ママと話してもいい？")
    assert passed is False and req is not None


def test_a_near_name_is_not_asked():
    """1 字違いの名前（テレビや崩れた書き起こし）では、入口が名前で呼ばれたと決めないので聞かない。"""
    jev = _Jev(0.99)
    ip = _silenced(jev)
    passed, req = _say(ip, "カジュ、もう怒ってないから声出して")
    assert passed is False and req is not None
    assert _release_calls(jev) == []


def test_the_name_alone_is_not_asked():
    jev = _Jev(0.99)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ")
    assert req is not None
    assert _release_calls(jev) == []


def test_if_the_request_changed_while_asking_it_is_not_lifted():
    """Jev を待つあいだに別の「黙って」が入ったら、その依頼は解かない。"""
    jev = _Jev(0.9)
    ip = _silenced(jev)
    ask = jev.ask

    async def ask_then_rehush(state, questions):
        got = await ask(state, questions)
        if "release" in questions:
            await ip._paju().hush_with(
                SilenceRequest(person="", until=time.time() + 3600, reason="人")
            )
        return got

    jev.ask = ask_then_rehush  # type: ignore[method-assign]
    passed, req = _say(ip, "パジュ、もう怒ってないから声出して")
    assert passed is False and req is not None


def test_without_jev_a_release_phrase_still_lifts_a_human_silence():
    """Jev が答えないときは、人の沈黙では字の形で決める（「パジュ、話していいよ」で戻れなくならない）。"""
    jev = _Jev(None)
    ip = _silenced(jev)
    passed, req = _say(ip, "パジュ、話していいよ")
    assert passed is True and req is None
