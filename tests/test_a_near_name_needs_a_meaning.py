"""1 字違いの名前は、言葉の意味を Jev に確かめてから窓を開ける（出-ch・2026-10-16・本人の決定サ）。

厳しい段（居ない・誰の声か分からない）は 1 字違いを許さず、崩れて書き起こされた本物の呼びかけ（パンシュー・カジュー・
パッシュ・バジル）を「名前が無い」で捨てていた。ふだんの段は 1 字違いを値も見ずに通すので、同じ 22 件なら呼んでいない
13 件のうち 10 件で窓が開いていた。ふつう名前のあとには用件が続く（本人）。

1 字違いでだけ当たったら、段にかかわらず発話の意味の一覧で Jev に聞き、「成立しない」「文脈に合わない」なら捨てる
（Jev が答えないときも）。厳しい段はそのうえでいまの「名前＋言葉」の値を見る。実機の 22 件で本物 4/4・誤り 0/13
（jev-1.13.0・`計測_設定値根拠台帳` §59）。
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


# ── Jev の判定 ──────────────────────────────────────────────────────────────


class _Jev:
    available = True

    def __init__(self, choice: str = "other", ok: bool = True):
        self._choice, self._ok = choice, ok
        self.calls: list = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        return SimpleNamespace(
            ok=self._ok,
            answers={"meaning": {"choice": self._choice, "confidence": 0.3}} if self._ok else {},
            model="jev-test",
        )


@pytest.mark.parametrize(
    "choice,ok,want",
    [
        ("other", True, True),
        ("time", True, True),
        ("unformed", True, False),
        ("off_context", True, False),
        ("other", False, False),
    ],
)
def test_the_meaning_decides(choice, ok, want):
    jev = _Jev(choice, ok)
    got = asyncio.run(jev_judges.judge_wake_meaning(jev, text="パッシュ3分測って"))
    assert got is want
    _state, qs = jev.calls[0]
    assert "unformed" in qs["meaning"]["criteria"]  # 選択肢は発話の意味の一覧


def test_no_jev_drops():
    assert asyncio.run(jev_judges.judge_wake_meaning(None, text="パッシュ3分測って")) is False


# ── 入口の門 ────────────────────────────────────────────────────────────────


def _gate(jev, *, here=False, known_voice=False):
    from tests.test_wake_strictness import _gate as base

    ip, a = base(here=here, known_voice=known_voice)
    a._jev = jev
    ip._wake_window().until = time.monotonic() - 60
    return ip


def _heard(ip, text, *, ns, lp):
    from tests.test_wake_strictness import _heard as base

    return base(ip, text, ns=ns, lp=lp)


@pytest.mark.real_window
def test_the_gate_asks_jev_for_a_near_name():
    jev = _Jev("other")
    ip = _gate(jev)
    assert _heard(ip, "パンシュー、おはよう。", ns=0.011, lp=-0.465) is True
    assert len(jev.calls) == 1


@pytest.mark.real_window
def test_the_gate_drops_a_near_name_that_means_nothing():
    jev = _Jev("unformed")
    ip = _gate(jev, here=True, known_voice=True)
    assert _heard(ip, "ジュシェイ", ns=0.044, lp=-0.720) is False


@pytest.mark.real_window
def test_the_gate_does_not_ask_for_the_exact_name():
    """**反証の側**：名前がそのままなら Jev を待たない。"""
    jev = _Jev("unformed")
    ip = _gate(jev)
    assert _heard(ip, "パジュ、おはよう", ns=0.011, lp=-0.465) is True
    assert jev.calls == []
