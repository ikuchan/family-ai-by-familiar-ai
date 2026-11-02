"""名前だけの声を仮の呼びかけとして預け、10 秒以内の次の言葉を宛先の確かめで受ける（知-bs・2026-11-02・本人の決定）。

11/02 12:53:50、「パジュ」とだけ呼んだ声は確かさ -0.893 で、名前だけの基準（-0.65 以上）に届かず窓を開けなかった。5 秒後の
20 字の問いは窓の外として捨てた。2 週間の記録（61 ファイル）で名前だけの呼びかけは 3 回、どれも開けず、2 回は続きの言葉も
捨てていた。名前だけで確かさの足りない声は、窓を開けず返事もせず、聞いた時刻を預ける。10 秒（`WAKE_WINDOW_SEC`）以内の
次の声は、捨てる前に「パジュ、」＋その言葉を宛先の判定（1 字違いの名前と同じ問いと下限）に見せ、宛てなら名前で呼ばれた
用件として受ける。宛てでない声（せき・テレビ）では消さず、10 秒のあいだ残す（見直しの指摘 3 番・本人の決定イ）。
宛てと分かって受けたときと、10 秒たったときに消える。
Jev の確かめ（`jev-1.14.0`・合成した場面・3 回）：頭に「パジュ、」を付けると、頼み 3 つはロボット宛て 0.73〜0.95、テレビや
家族の話 3 つは 0.01〜0.19。頭を付けないと、頼みは 0.33〜0.67 で下限 0.3 にすれすれだった。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from familiar_agent.loop.event_loop import Trigger
from tests.test_a_near_name_needs_a_meaning import _Jev, _gate

pytestmark = pytest.mark.real_window

T0 = time.monotonic() + 1000  # 名前だけの声が届いた時刻（`_gate` が閉じた窓より後）


def _say(ip, text, *, at, lp=-0.4, source="voice"):
    """届いた時刻 `at` の会話入力を門に通す。(trigger, 受けたか) を返す。"""

    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(
            kind="会話入力",
            query=text,
            future=fut,
            source=source,
            no_speech=0.02,
            logprob=lp,
            arrived=at,
        )
        return t, not await ip._swallow_if_unheard(t)

    return asyncio.run(go())


def _called_by_name_only(ip):
    """12:53:50 の形：名前だけで、確かさが足りない（-0.893）。窓は開かない。"""
    _t, ok = _say(ip, "パジュ", at=T0, lp=-0.893)
    assert ok is False and not ip._wake_window().is_open(T0)


@pytest.mark.parametrize("known_voice", [True, False])  # ふだんの段（家族の声）と厳しい段
def test_the_next_words_are_taken_when_they_are_for_paju(known_voice):
    jev = _Jev(0.9)
    ip = _gate(jev, known_voice=known_voice)
    _called_by_name_only(ip)
    t, ok = _say(ip, "明日の天気は？", at=T0 + 5)
    assert ok is True and t.named is True
    assert ip._wake_window().is_open(T0 + 5)
    assert t.window_session == ip._wake_window().session
    state, _qs = jev.calls[0]
    assert "パジュ、明日の天気は？" in state  # 頭に名前を付けて見せる


def test_words_not_for_paju_are_dropped():
    """**反証の側**：宛てでない（0.1）・Jev が答えないなら、いまどおり捨てる。"""
    for robot in (0.1, None):
        ip = _gate(_Jev(robot))
        _called_by_name_only(ip)
        _t, ok = _say(ip, "ごはんできたよー", at=T0 + 5)
        assert ok is False and not ip._wake_window().is_open(T0 + 5)


def test_just_before_ten_seconds_is_still_taken():
    jev = _Jev(0.9)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _t, ok = _say(ip, "明日の天気は？", at=T0 + 9.9)
    assert ok is True


def test_a_music_word_taken_while_playing_uses_up_the_hold():
    """**反証の側**：鳴っているあいだの音楽の言葉を受けたら（パジュ宛て）、預かりは消える。後のテレビの声に使わない。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    _called_by_name_only(ip)
    ip._music_word_while_playing = lambda trigger: trigger.query == "ストップ"
    _t, ok = _say(ip, "ストップ", at=T0 + 2)
    assert ok is True and ip._tentative_call is None


def test_after_ten_seconds_jev_is_not_asked():
    """**反証の側**：10 秒を過ぎた言葉は、Jev に聞かずに捨てる。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _t, ok = _say(ip, "明日の天気は？", at=T0 + 10)  # 窓と同じく、ちょうど 10 秒は外
    assert ok is False and jev.calls == []


def test_a_cough_in_between_does_not_lose_the_request():
    """宛てでない声（せき・テレビ）をはさんでも、10 秒のうちなら次の用件を受ける。受けたら預かりは消える。"""
    jev = _Jev(0.1)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _t, ok = _say(ip, "ん", at=T0 + 2)
    assert ok is False
    jev._robot = 0.9
    t, ok = _say(ip, "明日の天気は？", at=T0 + 6)
    assert ok is True and t.named is True and len(jev.calls) == 2
    assert ip._tentative_call is None


def test_a_name_that_opened_the_window_is_not_held():
    """**反証の側**：確かさが足りて窓を開けた名前だけの声は、いまの道のまま（窓の中の言葉は宛先の判定を通る）。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    _t, ok = _say(ip, "パジュ", at=T0, lp=-0.5)
    assert ok is False and ip._wake_window().is_open(T0)
    assert ip._tentative_call is None


def test_typed_words_do_not_use_the_hold():
    """**反証の側**：キーボードの入力は預かりを使わない（名前が無ければ窓の外）。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _t, ok = _say(ip, "明日の天気は？", at=T0 + 5, source="keyboard")
    assert ok is False and jev.calls == []


def test_a_dropped_name_repeat_keeps_the_hold():
    """先で捨てる声（名前の繰り返し）をはさんでも、10 秒のうちなら次の用件を受ける。"""
    jev = _Jev(0.9)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _say(ip, "パジュ、パジュー", at=T0 + 2)
    _t, ok = _say(ip, "明日の天気は？", at=T0 + 4)
    assert ok is True and len(jev.calls) == 1


def test_a_word_with_a_name_is_judged_by_the_name_rule_only():
    """**反証の側**：名前（1 字違い）の聞こえた声は名前の基準で決め、宛先を 2 度聞かない。"""
    jev = _Jev(0.1)
    ip = _gate(jev)
    _called_by_name_only(ip)
    _t, ok = _say(ip, "パンシュー、ごはんできたよ", at=T0 + 4)
    assert ok is False and len(jev.calls) == 1


# ── 1 字違いの名前だけの声も預ける（知-bs の続き・2026-11-02・本人の決定ア）─────────────────────
#
# 10 月の録音では、名前を Whisper に渡さないと「アジュー」などの 1 字違いで取れた（知-z-は）。名前だけの言葉には Jev が宛先を
# 決める材料がほとんど無いので聞かず、完全な名前と同じく預ける。窓は開けず、返事もしない。


@pytest.mark.parametrize("heard", ["アジュー", "ジュ"])
def test_a_near_name_alone_is_held_without_asking(heard):
    jev = _Jev(0.9)
    ip = _gate(jev)
    _t, ok = _say(ip, heard, at=T0, lp=-0.893)
    assert ok is False and jev.calls == [] and not ip._wake_window().is_open(T0)
    t, ok = _say(ip, "明日の天気は？", at=T0 + 5)
    assert ok is True and t.named is True and len(jev.calls) == 1


def test_a_near_name_with_words_is_still_asked():
    """**反証の側**：1 字違いの名前のあとに言葉が続けば、いまどおり宛先を聞き、預けない。"""
    jev = _Jev(0.1)
    ip = _gate(jev)
    _t, ok = _say(ip, "パンシュー、おはよう。", at=T0)
    assert ok is False and len(jev.calls) == 1 and ip._tentative_call is None


def test_a_near_name_alone_in_an_open_window_gets_no_reply():
    """**反証の側**：窓が開いていても、1 字違いの名前だけの声は返事に回さない（窓も延ばさない）。"""
    jev = _Jev(0.9)
    ip = _gate(jev, open_window=True)
    until = ip._wake_window().until
    _t, ok = _say(ip, "アジュー", at=time.monotonic())
    assert ok is False and jev.calls == [] and ip._wake_window().until == until
    assert ip._tentative_call is None  # 窓の中なので預けない


def test_a_near_name_with_more_sounds_is_asked_and_not_held():
    """**反証の側**：1 字違いの名前の後ろに音が続く声（「ジュース」）は名前だけではない。いまどおり宛先を聞き、預けない。"""
    jev = _Jev(0.1)
    ip = _gate(jev)
    _t, ok = _say(ip, "ジュース", at=T0)
    assert ok is False and len(jev.calls) == 1 and ip._tentative_call is None
