"""1 字違いの名前は、宛先を Jev に確かめてから窓を開ける（出-ch・2026-10-16・出-db 段 D・2026-10-25・本人の決定 3-c）。

厳しい段（居ない・誰の声か分からない）は 1 字違いを許さず、崩れて書き起こされた本物の呼びかけ（パンシュー・カジュー・
パッシュ・バジル）を「名前が無い」で捨てていた。ふだんの段は 1 字違いを値も見ずに通すので、同じ 22 件なら呼んでいない
13 件のうち 10 件で窓が開いていた。ふつう名前のあとには用件が続く（本人）。

1 字違いでだけ当たったら、段にかかわらず Jev に聞く。厳しい段はそのうえでいまの「名前＋言葉」の値を見る。以前は発話の意味
の一覧で聞き、「成立しない」「文脈に合わない」なら捨てていた（出-ch サ）が、言葉だけでは本物と崩れた声が分かれなかった。
出-db 段 D から、窓が閉じているときは宛先の判定（ロボット宛て 0.3 以上で開ける・Jev が答えなければ捨てる）で聞く。窓が
開いているときの 1 字違いは、名前の無い言葉と同じに宛先の判定を通す（聞き違えた名前の直しだけは行う）。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from familiar_agent.core import jev_judges
from familiar_agent.core import wake_strictness as ws

NAMES = ["パジュ"]


def _ok(text, *, strict, near_ok, ns, lp):
    return ws.admits(text, NAMES, strict=strict, no_speech=ns, logprob=lp, near_ok=near_ok)


# ── 純関数 ──────────────────────────────────────────────────────────────────


def test_a_near_name_with_a_meaning_passes_when_strict():
    assert _ok("パンシュー、おはよう。", strict=True, near_ok=True, ns=0.011, lp=-0.465)[0]


def test_a_near_name_without_a_meaning_is_dropped():
    """**反証の側**：同じ言葉でも、意味が成り立たなければ捨てる。"""
    ok, why = _ok("パンシュー、おはよう。", strict=True, near_ok=False, ns=0.011, lp=-0.465)
    assert not ok and "1 字違い" in why


def test_strict_still_reads_the_values():
    ok, why = _ok("マシュマロまでいったのに。", strict=True, near_ok=True, ns=0.434, lp=-0.677)
    assert not ok and "確かさが足りない" in why


def test_lenient_no_longer_passes_a_near_name_by_itself():
    assert not _ok("シュー", strict=False, near_ok=False, ns=0.370, lp=-0.938)[0]
    assert _ok("パチュ、いま何時？", strict=False, near_ok=True, ns=0.05, lp=-0.5)[0]


def test_the_exact_name_does_not_need_a_meaning():
    """**反証の側**：名前がそのまま当たるときは、意味を見ない（いままでどおり）。"""
    assert _ok("パジュ、いま何時？", strict=True, near_ok=False, ns=0.05, lp=-0.5)[0]
    assert ws.near_name_only("パンシュー、おはよう。", NAMES)
    assert not ws.near_name_only("パジュ、おはよう", NAMES)
    assert not ws.near_name_only("おはよう", NAMES)


# ── 窓が閉じているとき：宛先の判定で開ける ────────────────────────────────


class _Jev:
    available = True

    def __init__(self, robot: "float | None" = 0.9):
        self._robot = robot
        self.calls: list = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        if self._robot is None:
            return SimpleNamespace(ok=False, answers={}, model="jev-test")
        rest = (1.0 - self._robot) / 3
        probs = {"robot": self._robot, "family": rest, "nowhere": rest, "unknown": rest}
        top = max(probs, key=lambda k: probs[k])
        got = {"choice": top, "confidence": probs[top], "probabilities": probs}
        return SimpleNamespace(ok=True, answers={"to_whom": got}, model="jev-test")


def _gate(jev, *, here=False, known_voice=False, open_window=False):
    from tests.test_wake_strictness import _gate as base

    ip, a = base(here=here, known_voice=known_voice)
    a._jev = jev
    ip._wake_window().until = time.monotonic() + (60 if open_window else -60)
    return ip


def _heard(ip, text, *, ns, lp):
    from tests.test_wake_strictness import _heard as base

    return base(ip, text, ns=ns, lp=lp)


def _trigger_after_gate(ip, text):
    from familiar_agent.loop.event_loop import Trigger

    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(
            kind="会話入力", query=text, future=fut, source="voice", no_speech=0.01, logprob=-0.4
        )
        dropped = await ip._swallow_if_unheard(t)
        return t, dropped

    return asyncio.run(go())


@pytest.mark.real_window
@pytest.mark.parametrize(("robot", "want"), [(0.9, True), (0.29, False), (None, False)])
def test_a_near_name_opens_the_window_by_the_addressee(robot, want):
    """ロボット宛て 0.3 以上で開ける。0.3 未満と、Jev が答えないときは捨てる（崩れた音で窓を開けない）。"""
    jev = _Jev(robot)
    ip = _gate(jev)
    assert _heard(ip, "パンシュー、おはよう。", ns=0.011, lp=-0.465) is want
    _state, qs = jev.calls[0]
    assert "to_whom" in qs


@pytest.mark.real_window
def test_the_gate_does_not_ask_for_the_exact_name():
    """**反証の側**：名前がそのままなら Jev を待たない。"""
    jev = _Jev(0.0)
    ip = _gate(jev)
    assert _heard(ip, "パジュ、おはよう", ns=0.011, lp=-0.465) is True
    assert jev.calls == []


def test_the_meaning_judge_is_gone():
    assert not hasattr(jev_judges, "judge_wake_meaning")


# ── 窓が開いているとき：名前の無い言葉と同じに宛先の判定を通す ─────────────


@pytest.mark.real_window
def test_an_open_window_near_name_is_judged_like_an_unnamed_word():
    jev = _Jev(0.1)
    ip = _gate(jev, open_window=True)
    t, dropped = _trigger_after_gate(ip, "パンシュー、おはよう。")
    assert dropped and len(jev.calls) == 1
    assert not t.named and t.near_name


@pytest.mark.real_window
def test_an_open_window_near_name_that_is_for_me_is_heard_unnamed():
    jev = _Jev(0.9)
    ip = _gate(jev, open_window=True)
    t, dropped = _trigger_after_gate(ip, "パンシュー、おはよう。")
    assert not dropped and not t.named and t.near_name


@pytest.mark.real_window
def test_an_open_window_exact_name_is_still_named():
    """**反証の側**：窓が開いていても、名前がそのままなら名前つきで、Jev を待たない。"""
    jev = _Jev(0.0)
    ip = _gate(jev, open_window=True)
    t, dropped = _trigger_after_gate(ip, "パジュ、おはよう")
    assert not dropped and t.named and not t.near_name and jev.calls == []


@pytest.mark.real_window
def test_an_open_window_near_name_is_still_fixed():
    """名前つきには数えないが、聞き違えた名前の直し（知-ap）は行う（出-db 段 D・本人の決定ア）。"""
    from tests.test_addressee_with_jev import _answer, _ip, _push

    ip, a, _ = _ip(_answer("robot", 0.9))
    shown: list = []
    ip.set_heard_listener(lambda text, ok: shown.append(text))
    ip._wake_window().open(100.0)
    assert _push(ip, "パンシュ、明日の天気は？", arrived=105.0) is True
    assert a._jev.ask.await_count == 1  # 名前の無い言葉と同じに宛先を聞いた
    assert shown and "（パジュ）" in shown[-1]


@pytest.mark.real_window
def test_a_closed_window_near_name_sends_no_old_talk():
    """窓が閉じたなら会話は終わっている。直前のやりとりは送らない（終わった話の続きと読まれないため）。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    seen: dict = {}
    real = ip._ask_addressee

    async def spy(trigger, *, recent, drop_unanswered=False):
        seen.update(recent=recent, drop=drop_unanswered)
        return await real(trigger, recent=recent, drop_unanswered=drop_unanswered)

    ip._ask_addressee = spy
    assert _heard(ip, "パンシュー、おはよう。", ns=0.011, lp=-0.465) is True
    assert seen == {"recent": "", "drop": True}
    state, _ = jev.calls[0]
    assert "[直前のやりとり（古い順）]\n（なし）" in state


@pytest.mark.real_window
def test_an_open_window_near_name_is_heard_when_jev_does_not_answer():
    """**反証の側**：窓の中の 1 字違いは名前の無い言葉と同じなので、Jev が答えなければ受ける（捨てるのは窓が閉じているとき）。"""
    ip = _gate(_Jev(None), open_window=True)
    t, dropped = _trigger_after_gate(ip, "パンシュー、おはよう。")
    assert not dropped and not t.named


@pytest.mark.real_window
def test_an_open_window_near_name_does_not_abort_lookups():
    """名前つきに数えないので、調べかけを打ち切らない（本人の決定ア）。"""
    from tests.test_addressee_with_jev import _answer, _ip, _push

    ip, _a, _ = _ip(_answer("robot", 0.9))
    ip._wake_window().open(100.0)
    assert _push(ip, "パンシュ、明日の天気は？", arrived=105.0) is True
    ip._abort_lookups.assert_not_awaited()
    ip2, _a2, _ = _ip(_answer("robot", 0.9))
    ip2._wake_window().open(100.0)
    assert _push(ip2, "パジュ、明日の天気は？", arrived=105.0) is True
    ip2._abort_lookups.assert_awaited()  # **反証の側**：名前がそのままなら打ち切る
