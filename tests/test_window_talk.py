"""窓の会話（環-ar 段 A・2026-10-25・本人の決定）。

窓が開いてから閉じるまでの会話を、ひとつの会話として 1 行ずつ持つ。AI に渡すのは数と時間で切った直近ではなく、この
窓の会話の全部である。行は出来事が起きたその場で足し、書いた行は書き換えない（前の行が 1 字でも変わると、そこから後ろの
プロンプトキャッシュが外れる）。調べた結果は短い 1 行（80 字）で、全文はいまどおり作業状態の W に置く。
"""

from __future__ import annotations

from datetime import datetime

from familiar_agent.core import window_talk as wt
from familiar_agent.core.wake_window import WakeWindow


def _at(h: int, m: int, s: int) -> datetime:
    return datetime(2026, 10, 25, h, m, s)


# --- 窓の会話の番号と開いた時刻 ---


def test_opening_a_closed_window_starts_a_new_talk():
    w = WakeWindow()
    w.open(100.0)
    first = w.session
    assert first >= 1 and w.started == 100.0
    w.open(105.0)  # 開いているあいだに名前で呼んでも、同じ会話
    w.extend(108.0)
    assert (w.session, w.started) == (first, 100.0)
    w.open(200.0)  # 閉じた後に開く
    assert w.session == first + 1 and w.started == 200.0


def test_reopening_for_a_reply_continues_the_talk():
    """返事の話し終わりに開け直すとき（出-cx・出-cw）は、同じ会話の続き。"""
    w = WakeWindow()
    w.open(100.0)
    first = w.session
    w.open(150.0, resume=True)
    assert (w.session, w.started) == (first, 100.0)


def test_a_resume_with_no_talk_yet_starts_one():
    w = WakeWindow()
    w.open(10.0, resume=True)
    assert w.session == 1 and w.started == 10.0


# --- 行 ---


def test_lines_are_kept_in_order_and_never_rewritten():
    talk = wt.WindowTalk(session=1, opened_at=_at(19, 0, 0))
    talk.add(_at(19, 0, 0), "パパ", "パジュ、しりとりしよう")
    before = talk.render()
    talk.add(_at(19, 0, 2), "わたし（パパへ）", "いいよ。しりとり！")
    after = talk.render()
    assert after.startswith(before)  # 前の行は 1 字も変わらない
    assert after.splitlines() == [
        "[窓の会話（19:00:00 から・古い順・全部）]",
        "- 19:00:00 パパ：パジュ、しりとりしよう",
        "- 19:00:02 わたし（パパへ）：いいよ。しりとり！",
    ]


def test_blocks_are_the_header_and_each_line():
    talk = wt.WindowTalk(session=1, opened_at=_at(19, 0, 0))
    talk.add(_at(19, 0, 0), "パパ", "りす")
    talk.add(_at(19, 0, 1), "わたし", "すいか！")
    assert talk.blocks() == (
        "[窓の会話（19:00:00 から・古い順・全部）]",
        "- 19:00:00 パパ：りす",
        "- 19:00:01 わたし：すいか！",
    )
    assert "\n".join(talk.blocks()) == talk.render()


def test_an_empty_talk_renders_nothing():
    talk = wt.WindowTalk(session=1, opened_at=_at(19, 0, 0))
    assert talk.render() == "" and talk.blocks() == ()


def test_a_result_is_one_short_line():
    line = wt.result_text("明日の天気", "晴れ。" + "最高 18 度、" * 40)
    assert line.startswith("調べた（明日の天気）：晴れ。")
    assert len(line) <= len("調べた（明日の天気）：") + wt.RESULT_MAX_CHARS + 1
    assert wt.RESULT_MAX_CHARS == 80


def test_newlines_inside_words_stay_on_one_line():
    talk = wt.WindowTalk(session=1, opened_at=_at(19, 0, 0))
    talk.add(_at(19, 0, 0), "わたし", "一行目\n二行目")
    assert talk.blocks()[1] == "- 19:00:00 わたし：一行目 二行目"
