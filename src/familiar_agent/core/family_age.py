"""年齢と学年を、誕生日と今日の日付から数える（知-ad 段 1・2026-10-01）。純関数。

`FAMILY.md` は人の入力のまま、機械は書き換えない（開発ルール「ファイルは既定値と人の入力だけ」・本人の決定イ）。
年齢は毎年ずれ、学年は 4 月に上がるので、書かれた年齢ではなく**誕生日から毎回数えて**システム文に添える。
保存しないので古くならない。

学年は日本の学校の区切りで数える：4 月 2 日生まれから翌年 4 月 1 日生まれまでが同じ学年（4 月 1 日生まれは
法律上 3 月 31 日に歳をとるので、上の学年）。年度は 4 月 1 日に始まる。小学 1 年〜高校 3 年だけを言葉にする。
"""

from __future__ import annotations

import re
from datetime import date

from .parsing import parse_family_md

_JA = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_TEMPLATE = re.compile(r"^[（(].*[）)]$")

#: 学年の番号（小 1 が 1）→ 言葉。この外（未就学・高校の後）は添えない。
_GRADES = {
    **{n: f"小学 {n} 年" for n in range(1, 7)},
    **{n + 6: f"中学 {n} 年" for n in range(1, 4)},
    **{n + 9: f"高校 {n} 年" for n in range(1, 4)},
}

HEADING = "[いまの年齢と学年（誕生日と今日の日付から数えた）]"


def parse_birthday(text: str) -> "date | None":
    """「1980 年 3 月 4 日」か「1980-03-04」を読む。雛形の括弧書きや読めないものは None。"""
    s = (text or "").strip()
    if not s or _TEMPLATE.match(s):
        return None
    m = _JA.search(s) or _ISO.search(s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def age_on(born: date, today: date) -> int:
    """満年齢（誕生日に 1 つ上がる）。"""
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def grade_on(born: date, today: date) -> "str | None":
    """その日の学年。小学 1 年〜高校 3 年の外なら None。"""
    cohort = born.year if (born.month, born.day) >= (4, 2) else born.year - 1
    school_year = today.year if today.month >= 4 else today.year - 1
    return _GRADES.get(school_year - cohort - 6)


def ages_lines(family_md: str, today: date) -> "list[str]":
    """家族ひとりずつ「呼び方：N 歳・学年」。誕生日の無い・読めない人は飛ばす。"""
    lines: list[str] = []
    for m in parse_family_md(family_md or ""):
        born = parse_birthday(str(m.get("birthday") or ""))
        if born is None or born > today:
            continue
        who = str(m.get("display_name") or m.get("name") or "").split("、")[0].strip()
        grade = grade_on(born, today)
        lines.append(f"{who}：{age_on(born, today)} 歳" + (f"・{grade}" if grade else ""))
    return lines


def render(family_md: str, today: date) -> str:
    """`[いまの年齢と学年]` の枠。誰も数えられなければ空。"""
    lines = ages_lines(family_md, today)
    return HEADING + "\n" + "\n".join(lines) if lines else ""
