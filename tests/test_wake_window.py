"""ウェイクワードと 1 分の窓（出-as 段 1・2026-09-26・`設計方針_話していいかの決まり` v0.1 §2.3）。

名前を聞いたら 1 分の窓を開く。窓の中の入力・返事・つなぎで、そこから 1 分へ延ばす。窓の外の声は会話に
しない。判定は、書き起こしを直した後の文の**文頭に** `ME.md` の名前があるか（1 字違いまで・黙る依頼の
`names_me` と同じゆるさ）。2026-09-30 に「文のどこか」から改めた（本人の決定・`test_the_name_comes_first`）。窓は出来事で開け閉めし、声が鳴ったか・マイクで聞いたかは見ない。
"""

from __future__ import annotations

from familiar_agent.core.wake_window import WAKE_WINDOW_SEC, WakeWindow, heard_name

NAMES = ["パジュ"]


# ── ウェイクワード ─────────────────────────────────────────────────────────


def test_the_name_at_the_head_is_heard():
    assert heard_name("パジュ、30分黙ってて", NAMES)
    assert heard_name("パジュは今日元気？", NAMES)
    assert not heard_name("明日の天気、パジュに聞いてみよう", NAMES)  # 文の途中の名前は聞かない


def test_one_character_off_is_heard():
    """1 字違いまで（「パジュー」「バジュ」）。直しの後に残った揺れを拾う。"""
    assert heard_name("パジュー、パジュー", NAMES)
    assert heard_name("バジュ、聞こえる？", NAMES)


def test_other_talk_is_not_heard():
    assert not heard_name("ごちそうさまでした", NAMES)
    assert not heard_name("じゃあ俺がビオネットね", NAMES)


def test_without_names_nothing_is_heard():
    """名前が設定されていなければ、声は何も聞かない（本人の決定 2026-09-26）。キーボードは別の口で受ける。"""
    assert not heard_name("ごちそうさまでした", [])
    assert not heard_name("パジュ、聞こえる？", [])


# ── 窓 ────────────────────────────────────────────────────────────────────


def test_the_window_is_ten_seconds():
    """出-as の 1 分 → 30 秒（出-au）→ 10 秒（2026-10-07 本人の決定「窓の時間は１０秒で」）。"""
    assert WAKE_WINDOW_SEC == 10.0
    w = WakeWindow()
    assert not w.is_open(100.0)
    w.open(100.0)
    assert w.is_open(109.9)
    assert not w.is_open(110.0)


def test_extending_inside_the_window_restarts_it():
    w = WakeWindow()
    w.open(100.0)
    w.extend(108.0)  # 窓の中で入力・返事・つなぎ
    assert w.is_open(117.9) and not w.is_open(118.0)


def test_extending_after_it_closed_does_nothing():
    """窓が切れた後の返事は話さない（独り言）。延ばして開け直さない。"""
    w = WakeWindow()
    w.open(100.0)
    w.extend(111.0)
    assert not w.is_open(111.0)


def test_opening_again_never_shortens():
    w = WakeWindow()
    w.open(100.0)
    w.extend(108.0)
    w.open(102.0)  # 古い時刻で開けても縮まない
    assert w.is_open(117.9)


def test_close_shuts_it():
    w = WakeWindow()
    w.open(100.0)
    w.close()
    assert not w.is_open(101.0)


# ── 話しているあいだ（2026-10-07 実機 23:18：話しているうちに窓が切れ、聞き返しへの返事を 3 回捨てた）────


def test_the_window_stays_open_while_speaking():
    """話し始めに開いていれば、話しているあいだは `until` を過ぎても開いている。"""
    w = WakeWindow()
    w.open(100.0)  # 110 まで
    w.hold()
    assert w.is_open(115.0)


def test_it_closes_ten_seconds_after_speaking_ends():
    w = WakeWindow()
    w.open(100.0)
    w.hold()
    w.release(116.0)  # 話し終わり
    assert w.is_open(125.9) and not w.is_open(126.0)


def test_two_voices_keep_it_open_until_both_end():
    """つなぎと本応答が重なっても、両方が話し終わるまで開いている。"""
    w = WakeWindow()
    w.open(100.0)
    w.hold()
    w.hold()
    w.release(112.0)
    assert w.is_open(130.0)
    w.release(114.0)
    assert w.is_open(123.9) and not w.is_open(124.0)


def test_close_ends_the_hold_too():
    w = WakeWindow()
    w.open(100.0)
    w.hold()
    w.close()
    assert not w.is_open(101.0)
