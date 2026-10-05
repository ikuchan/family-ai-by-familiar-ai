"""名前で会話の窓を開ける基準（2026-10-05 実機・本人の決定・`設計方針_話していいかの決まり` §2.3）。

実機 2026-10-05 23:34、誰も居ない部屋の音（1.1 秒）を書き起こしが「パジュー」と聞き違え（無音らしさ
`no_speech_prob` 0.150・確かさ `avg_logprob` -0.723）、文頭の名前で窓が開いて返事をした。

基準を 2 段にする（どれも仮の値）。

- **名前だけ**（名前 1 つと、伸ばす音・句読点だけの書き起こし）：ふだん 無音らしさ 0.10 以下・確かさ -0.65 以上、
  厳しい 0.05 以下・-0.60 以上で受ける。名前のくり返し（「パジュ、パジュー」）は名前＋言葉の側に入る。
- **名前＋言葉**：ふだんはいまのまま（1 字違いを許す・確かさを見ない）。厳しいときは 1 字違いを許さず、無音らしさ
  0.20 以下・確かさ -0.90 以上で受ける。
- 確かさが無い入力（ElevenLabs・TUI）は確かさを見ずに通す。
"""

from __future__ import annotations

import pytest

from familiar_agent.core import wake_strictness as ws

NAMES = ["パジュ"]


def _ok(text, *, strict, ns=None, lp=None):
    return ws.admits(text, NAMES, strict=strict, no_speech=ns, logprob=lp)[0]


@pytest.mark.parametrize(
    "text,want",
    [
        ("パジュー", True),
        ("パジュ！", True),
        ("、パジュ。", True),
        ("ぱじゅう", True),
        ("パジュ、パジュー", False),  # くり返しは名前＋言葉の側
        ("パジュ、いま何時？", False),
        ("おはよう", False),
    ],
)
def test_name_only(text, want):
    assert ws.is_name_only(text, NAMES) is want


@pytest.mark.parametrize("strict", [False, True])
def test_tonights_mishearing_is_dropped(strict):
    assert not _ok("パジュー", strict=strict, ns=0.150, lp=-0.723)


@pytest.mark.parametrize("strict", [False, True])
def test_a_real_call_on_0926_is_admitted(strict):
    assert _ok("パジュ", strict=strict, ns=0.019, lp=-0.558)


def test_a_name_only_between_the_two_levels():
    assert _ok("パジュ", strict=False, ns=0.08, lp=-0.62)  # ふだんは受ける
    assert not _ok("パジュ", strict=True, ns=0.08, lp=-0.62)  # 厳しいときは捨てる


@pytest.mark.parametrize(
    "text,ns,lp",
    [
        ("パジュ、パジュー", 0.120, -0.702),
        ("パジュ、パジュ、パジュ", 0.096, -0.571),
        ("パジューにできないの", 0.031, -0.824),
        ("パジュ、パジュー", 0.129, -0.846),
        ("パジュ、パジュ、パジュ", 0.123, -0.601),
        ("パジュ、パジュー", 0.105, -0.758),
        ("パジュう、パジュう、パジュう", 0.114, -0.639),
        ("パジュ、いま何時？", 0.044, -0.421),
    ],
)
def test_real_calls_with_words_pass_even_when_strict(text, ns, lp):
    assert _ok(text, strict=True, ns=ns, lp=lp)


def test_strict_does_not_allow_one_letter_off():
    assert _ok("パチュ、いま何時？", strict=False, ns=0.05, lp=-0.5)  # 1 字違い（ふだんは許す）
    assert not _ok("パチュ、いま何時？", strict=True, ns=0.05, lp=-0.5)


def test_strict_words_need_the_floor():
    assert not _ok("パジュ、いま何時？", strict=True, ns=0.25, lp=-0.5)
    assert not _ok("パジュ、いま何時？", strict=True, ns=0.05, lp=-0.95)
    assert _ok("パジュ、いま何時？", strict=False, ns=0.25, lp=-0.95)  # ふだんは確かさを見ない


def test_no_measurements_pass():
    assert _ok("パジュー", strict=True)  # ElevenLabs・TUI は測りようがない


def test_without_the_name_nothing_passes():
    assert not _ok("おはよう", strict=False, ns=0.0, lp=-0.1)


def test_the_reason_is_told():
    ok, why = ws.admits("パジュー", NAMES, strict=True, no_speech=0.150, logprob=-0.723)
    assert not ok and "名前だけ" in why
