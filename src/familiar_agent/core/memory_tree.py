"""記憶の木の節（記-m・2026-10-01）。純関数。

REST の要約は「人 → 年 → 月 → 日」の木になる。節は日（YYYY-MM-DD）・月（YYYY-MM）・年（YYYY）で書き、
その節の要約は、要約の時刻（その日・月・年の終わり）が入る範囲で引く。範囲は **UTC の暦**で切る——
日ごとの要約は日付の文字列を UTC の 23:59 に置いて書いている（`rest_fold._day_end`）ので、同じ境目で引く。

家族全体と人ごとは**種類で分ける**（人ごとはその人の面から引く）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

_NODE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")

#: 節の段ごとの種類（家族全体・人ごと）。向きとの対応は `io/oif._KIND_OF_DIRECTION`。
_KINDS = {
    ("日", False): ("day_summary",),
    ("月", False): ("month_summary",),
    ("年", False): ("year_summary",),
    ("日", True): ("person_summary",),
    ("月", True): ("person_month_summary",),
    ("年", True): ("person_year_summary",),
}


def parse_node(text: str) -> "tuple[str, datetime, datetime] | None":
    """節の書き方を読む。(段, 始まり, 終わり) を返す（終わりは含まない）。読めなければ None。"""
    m = _NODE.match((text or "").strip())
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    try:
        if d is not None:
            start = datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
            nxt = datetime.fromordinal(start.toordinal() + 1).replace(tzinfo=timezone.utc)
            return "日", start, nxt
        if mo is not None:
            start = datetime(int(y), int(mo), 1, tzinfo=timezone.utc)
            nxt = datetime(int(y) + (int(mo) == 12), int(mo) % 12 + 1, 1, tzinfo=timezone.utc)
            return "月", start, nxt
        return (
            "年",
            datetime(int(y), 1, 1, tzinfo=timezone.utc),
            datetime(int(y) + 1, 1, 1, tzinfo=timezone.utc),
        )
    except ValueError:
        return None


def kinds_for(level: str, *, person: bool) -> "tuple[str, ...]":
    """その段の要約の種類。人を指していれば人ごとのもの。"""
    return _KINDS[(level, bool(person))]


#: 1 つ下の段。暦のまとめ（REST 層 1 の ③）の材料は、1 つ下の段の要約である。
CHILD = {"年": "月", "月": "日"}
