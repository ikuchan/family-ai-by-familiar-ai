"""年齢と学年は、誕生日と今日の日付から毎回数える（知-ad 段 1・2026-10-01・本人の決定イ）。

`FAMILY.md` の年齢は毎年ずれ、学年は 4 月に上がる。人が書き直さなければ、パジュは去年の子どもとして
話し続ける。`FAMILY.md` は人の入力のまま機械は書き換えず（開発ルール）、システム文を組むたびに誕生日から
数えて `[一緒に暮らす人たち]` の後ろへ添える。保存しないので古くならない。学年は日本の学校の区切り
（4 月 2 日生まれ〜翌年 4 月 1 日生まれが同じ学年）で、小学 1 年〜高校 3 年に当たる人にだけ添える。
"""

from __future__ import annotations

from datetime import date

import pytest

from familiar_agent.core.context_parts import Stance, build_context
from familiar_agent.core.family_age import age_on, ages_lines, grade_on, parse_birthday

FAMILY = """## パパ

- **名前**：雄輔
- **呼び方**：パパ、ゆうすけ
- **誕生日**：1980 年 3 月 4 日

## たいき

- **名前**：泰輝
- **呼び方**：たいき
- **誕生日**：2014-04-02

## ばあば

- **名前**：花子
- **呼び方**：ばあば
"""


@pytest.mark.parametrize(
    "text,want",
    [
        ("1980 年 3 月 4 日", date(1980, 3, 4)),
        ("1980年3月4日", date(1980, 3, 4)),
        ("2014-04-02", date(2014, 4, 2)),
        ("（例：1980 年 3 月 4 日）", None),
        ("春ごろ", None),
        ("2014-02-30", None),
    ],
)
def test_a_birthday_is_read(text, want):
    assert parse_birthday(text) == want


def test_the_age_turns_on_the_birthday():
    born = date(1980, 3, 4)
    assert age_on(born, date(2026, 3, 3)) == 45
    assert age_on(born, date(2026, 3, 4)) == 46


@pytest.mark.parametrize(
    "born,today,want",
    [
        # 4 月 1 日生まれは早生まれ（法律上 3 月 31 日に 6 歳になる）で、その 4 月に入学する
        (date(2019, 4, 1), date(2025, 4, 1), "小学 1 年"),
        (date(2019, 4, 2), date(2025, 4, 1), None),  # 4 月 2 日生まれは翌年度に入学
        (date(2019, 4, 2), date(2026, 4, 1), "小学 1 年"),
        (date(2019, 4, 1), date(2025, 3, 31), None),  # 年度の前日はまだ入学していない
        (date(2014, 4, 2), date(2026, 10, 1), "小学 6 年"),
        (date(2011, 4, 2), date(2026, 10, 1), "中学 3 年"),
        (date(2008, 4, 2), date(2026, 10, 1), "高校 3 年"),
        (date(2007, 4, 2), date(2026, 10, 1), None),  # 高校を出た
        (date(1980, 3, 4), date(2026, 10, 1), None),
    ],
)
def test_the_school_grade_follows_the_april_line(born, today, want):
    assert grade_on(born, today) == want


def test_the_lines_name_each_person_by_how_they_are_called():
    lines = ages_lines(FAMILY, date(2026, 10, 1))
    assert lines == ["パパ：46 歳", "たいき：12 歳・小学 6 年"]  # 誕生日の無いばあばには添えない


def test_the_family_frame_carries_the_ages():
    ctx = build_context(
        stance=Stance.PAJU, self_understanding="パジュ", family=FAMILY, today=date(2026, 10, 1)
    )
    family_at = ctx.stable.index("[一緒に暮らす人たち]")
    ages_at = ctx.stable.index("[いまの年齢と学年（誕生日と今日の日付から数えた）]")
    assert family_at < ages_at
    assert "たいき：12 歳・小学 6 年" in ctx.stable


def test_no_ages_frame_without_birthdays():
    ctx = build_context(
        stance=Stance.PAJU, self_understanding="パジュ", family="## ばあば\n- **名前**：花子\n"
    )
    assert "[いまの年齢と学年" not in ctx.stable
