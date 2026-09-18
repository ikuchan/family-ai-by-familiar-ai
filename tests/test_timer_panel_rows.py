"""GUI のパネルにタイマーの状況（環-p・2026-09-18）。

在席・話者パネルは PMM の表を 2 秒ごとに映すだけで、タイマー（残り／一時停止中／鳴っている／聞いていない）が
GUI から読めなかった（実機 21:5x〜22:1x）。`[タイマー]`／`[アラーム]` の枠（主LLM に渡す文）と `DIF.ringing` を
そのまま行にする。読むだけで状態は変えない。
"""

from __future__ import annotations

from familiar_agent.gui import format_timer_rows

TIMER = "[タイマー]\n- id=16 パパの依頼 残り 1:29\n- 聞いていない（タイマー「パパの依頼」が鳴るまで。止めて・一時停止・再開の言葉だけ届く）"
ALARM = "[アラーム]\n- id=2 起こす 明日 06:30 に鳴る（静かな時間でも鳴らす）"


def test_frames_become_rows_without_the_headings():
    rows = format_timer_rows(TIMER, ringing=False, alarm_frame="")
    assert rows == [
        "id=16 パパの依頼 残り 1:29",
        "聞いていない（タイマー「パパの依頼」が鳴るまで。止めて・一時停止・再開の言葉だけ届く）",
    ]


def test_ringing_comes_first_and_alarms_follow():
    rows = format_timer_rows(TIMER, ringing=True, alarm_frame=ALARM)
    assert rows[0] == "🔔 鳴っている"
    assert rows[-1] == "id=2 起こす 明日 06:30 に鳴る（静かな時間でも鳴らす）"


def test_nothing_gives_one_quiet_row():
    assert format_timer_rows("", ringing=False, alarm_frame="") == [
        "（タイマー・アラーム・ストップウォッチなし）"
    ]
