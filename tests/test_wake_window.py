"""ウェイクワードと 1 分の窓（出-as 段 1・2026-09-26・`設計方針_話していいかの決まり` v0.1 §2.3）。

名前を聞いたら 1 分の窓を開く。窓の中の入力・返事・つなぎで、そこから 1 分へ延ばす。窓の外の声は会話に
しない。判定は、書き起こしを直した後の文の**どこかに** `ME.md` の名前があるか（1 字違いまで・黙る依頼の
`names_me` と同じゆるさ）。窓は出来事で開け閉めし、声が鳴ったか・マイクで聞いたかは見ない。
"""

from __future__ import annotations

from familiar_agent.core.wake_window import WINDOW_SEC, WakeWindow, heard_name

NAMES = ["パジュ"]


# ── ウェイクワード ─────────────────────────────────────────────────────────


def test_the_name_anywhere_is_heard():
    assert heard_name("パジュ、30分黙ってて", NAMES)
    assert heard_name("パジュは今日元気？", NAMES)
    assert heard_name("明日の天気、パジュに聞いてみよう", NAMES)


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


def test_the_window_is_thirty_seconds():
    """出-as の 1 分から 30 秒へ（出-au・`設計方針_判定の段` §2.1）。"""
    assert WINDOW_SEC == 30.0
    w = WakeWindow()
    assert not w.is_open(100.0)
    w.open(100.0)
    assert w.is_open(129.9)
    assert not w.is_open(130.0)


def test_extending_inside_the_window_restarts_it():
    w = WakeWindow()
    w.open(100.0)
    w.extend(120.0)  # 窓の中で入力・返事・つなぎ
    assert w.is_open(149.9) and not w.is_open(150.0)


def test_extending_after_it_closed_does_nothing():
    """窓が切れた後の返事は話さない（独り言）。延ばして開け直さない。"""
    w = WakeWindow()
    w.open(100.0)
    w.extend(140.0)
    assert not w.is_open(140.0)


def test_opening_again_never_shortens():
    w = WakeWindow()
    w.open(100.0)
    w.extend(120.0)
    w.open(110.0)  # 古い時刻で開けても縮まない
    assert w.is_open(149.9)


def test_close_shuts_it():
    w = WakeWindow()
    w.open(100.0)
    w.close()
    assert not w.is_open(101.0)
