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
        ("パジューにできないの", 0.031, -0.824),
        ("パジュ、いま何時？", 0.044, -0.421),
    ],
)
def test_real_calls_with_words_pass_even_when_strict(text, ns, lp):
    assert _ok(text, strict=True, ns=ns, lp=lp)


# ── 名前だけの繰り返しは想定外（2026-10-07 本人の決定）────────────────────────
#
# 実機 2026-10-07 15:17〜15:41、誰も呼んでいないのに「パジュ、パジュ、パジュー」と書き起こされ（無音らしさ 0.08〜0.19・
# 確かさ -0.49〜-0.85）、パジュが返事をした。名前だけを繰り返す声は想定外として、声なら値に関わらず捨てる（本人：
# 「そもそも名前を繰り返すことは想定外にして、無視してください」）。以前は名前＋言葉の側に入れ、09-26 の本物の呼びかけ
# （下の 5 件）を通していたが、それも捨てる。名前のあとに言葉が続く形は、いまどおり文頭の名前で判定する。


@pytest.mark.parametrize(
    "text",
    [
        "パジュ、パジュー",
        "パジュ、パジュ、パジュ",
        "パジュう、パジュう、パジュう",
        "パジュ、パジュ、パジュ、パジュー",
        "パジュう、パジュう",
        "パジュパジュ",
    ],
)
def test_a_repeat_of_the_name_alone(text):
    assert ws.is_name_repeat(text, NAMES)


@pytest.mark.parametrize(
    "text",
    ["パジュー", "パジュ！", "パジュ、パジュ、いま何時？", "パジュ、いま何時？", "おはよう", ""],
)
def test_not_a_repeat(text):
    assert not ws.is_name_repeat(text, NAMES)


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


# ── 入口の門（段 3）──────────────────────────────────────────────────────────
#
# 窓が閉じているときの声だけに効く。厳しいのは、居ない（顔ぶれも在席も無い）とき、または誰の声か分からない（声の特徴が
# 無い・どの家族にも 0.35 以上で当たらない）とき。以前は居なくても、最後に窓が閉じてから 10 分たつまでは厳しくしなかった
# （2026-10-07 本人の決定で撤去：「在席がないときにはパジュといったかどうかを厳しく判定できたりしませんか？」）。


def _gate(*, here=False, known_voice=False):
    from tests.test_the_wake_window_gates_voice import _ip

    ip, a = _ip(present=1.0 if here else 0.0)
    a.config.stt = __import__("familiar_agent.config", fromlist=["STTConfig"]).STTConfig()
    ip._voice_unknown = lambda voice: not known_voice  # type: ignore[method-assign]
    return ip, a


def _heard(ip, text, *, source="voice", ns=None, lp=None, voice=None):
    import asyncio

    from familiar_agent.loop.event_loop import Trigger

    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(
            kind="会話入力",
            query=text,
            future=fut,
            source=source,
            no_speech=ns,
            logprob=lp,
            voice=voice,
        )
        return not await ip._swallow_if_unheard(t)

    return asyncio.run(go())


@pytest.mark.real_window
def test_tonights_form_is_dropped_at_the_gate():
    ip, _ = _gate()
    assert _heard(ip, "パジュー", ns=0.150, lp=-0.723) is False


@pytest.mark.real_window
def test_a_real_call_opens_the_window():
    ip, _ = _gate()
    assert _heard(ip, "パジュ", ns=0.019, lp=-0.558) is True


@pytest.mark.real_window
def test_an_open_window_still_hears_everything():
    import time

    ip, _ = _gate()
    ip._wake_window().open(time.monotonic())
    assert _heard(ip, "パジュー", ns=0.150, lp=-0.723) is True


@pytest.mark.real_window
def test_the_keyboard_is_unchanged():
    ip, _ = _gate()
    assert _heard(ip, "パジュー", source="keyboard") is True


@pytest.mark.real_window
def test_with_someone_here_a_known_voice_and_a_recent_talk_it_is_lenient():
    import time

    ip, _ = _gate(here=True, known_voice=True)
    ip._wake_window().until = time.monotonic() - 60  # 1 分前に閉じた
    assert _heard(ip, "パチュ、いま何時？", ns=0.05, lp=-0.5) is True  # 1 字違いを許す


@pytest.mark.real_window
def test_an_unknown_voice_makes_it_strict_even_with_someone_here():
    import time

    ip, _ = _gate(here=True, known_voice=False)
    ip._wake_window().until = time.monotonic() - 60
    assert _heard(ip, "パチュ、いま何時？", ns=0.05, lp=-0.5) is False


@pytest.mark.real_window
def test_nobody_here_makes_it_strict_at_once():
    import time

    ip, _ = _gate(here=False, known_voice=True)
    ip._wake_window().until = time.monotonic() - 60  # 1 分前に閉じたばかりでも
    assert _heard(ip, "パチュ、いま何時？", ns=0.05, lp=-0.5) is False


def test_the_quiet_minutes_setting_is_gone():
    from familiar_agent.config import STTConfig

    assert not hasattr(STTConfig(), "wake_quiet_minutes")


def test_an_unknown_voice_is_one_without_a_feature_or_a_match(monkeypatch):
    import numpy as np

    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_voice_picks_the_speaker import _ip as _vip, _Store

    ip, a, s = _vip(store=_Store(registered={"papa": np.asarray([1.0, 0.0], dtype=np.float32)}))
    assert InformationProcessing._voice_unknown(ip, None) is True
    assert (
        InformationProcessing._voice_unknown(ip, np.asarray([1.0, 0.0], dtype=np.float32)) is False
    )
    assert (
        InformationProcessing._voice_unknown(ip, np.asarray([0.0, 1.0], dtype=np.float32)) is True
    )


@pytest.mark.real_window
@pytest.mark.parametrize("window_open", [False, True])
def test_a_repeated_name_is_dropped_at_the_gate(window_open, caplog):
    import time

    ip, _ = _gate(here=True, known_voice=True)
    if window_open:
        ip._wake_window().open(time.monotonic())
    with caplog.at_level("INFO"):
        assert _heard(ip, "パジュ、パジュ、パジュー", ns=0.083, lp=-0.513) is False
    assert "名前の繰り返し" in caplog.text


@pytest.mark.real_window
def test_a_repeated_name_typed_on_the_keyboard_is_kept():
    ip, _ = _gate()
    assert _heard(ip, "パジュ、パジュ", source="keyboard") is True


@pytest.mark.real_window
def test_a_repeat_followed_by_words_still_opens():
    ip, _ = _gate(here=True, known_voice=True)
    assert _heard(ip, "パジュ、パジュ、いま何時？", ns=0.05, lp=-0.5) is True
