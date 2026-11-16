"""記憶の木の節（記-m・2026-10-01）。純関数。

REST の要約は「人 → 年 → 月 → 日」の木になる。節は日（YYYY-MM-DD）・月（YYYY-MM）・年（YYYY）で書き、
その節の要約は、要約の時刻（その日・月・年の終わり）が入る範囲で引く。範囲は **ローカルの暦**で切る——
日の分け方（`rest_fold`）はローカルの日付で、要約はその日のローカル 23:59:59 か、その日のうちに書いたなら
書いた時刻で記録する（`materialize_save_event`）ので、同じ境目で引く（記-aa・2026-11-16。以前は UTC の暦で切り、
深夜 0〜9 時に書いた当日のまとめが前の日に入った）。

家族全体と人ごとは**向きで分ける**（人ごとはその人の面から引く・記-u 段 2 で種類から向きへ移した）。
"""

from __future__ import annotations

import re
from datetime import datetime

from ..store import clock

_NODE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")

#: 節の段ごとの要約の向き（家族全体・人ごと）。
_DIRECTIONS = {
    ("日", False): ("記憶",),
    ("月", False): ("月のまとめ",),
    ("年", False): ("年のまとめ",),
    ("日", True): ("人物",),
    ("月", True): ("人の月のまとめ",),
    ("年", True): ("人の年のまとめ",),
}


def parse_node(text: str) -> "tuple[str, datetime, datetime] | None":
    """節の書き方を読む。(段, 始まり, 終わり) を返す（終わりは含まない）。読めなければ None。"""
    m = _NODE.match((text or "").strip())
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    tz = clock.local_tz()
    try:
        if d is not None:
            start = datetime(int(y), int(mo), int(d), tzinfo=tz)
            nxt = datetime.fromordinal(start.toordinal() + 1).replace(tzinfo=tz)
            return "日", start, nxt
        if mo is not None:
            start = datetime(int(y), int(mo), 1, tzinfo=tz)
            nxt = datetime(int(y) + (int(mo) == 12), int(mo) % 12 + 1, 1, tzinfo=tz)
            return "月", start, nxt
        return (
            "年",
            datetime(int(y), 1, 1, tzinfo=tz),
            datetime(int(y) + 1, 1, 1, tzinfo=tz),
        )
    except ValueError:
        return None


def directions_for(level: str, *, person: bool) -> "tuple[str, ...]":
    """その段の要約の向き。人を指していれば人ごとのもの。"""
    return _DIRECTIONS[(level, bool(person))]


#: 1 つ下の段。暦のまとめ（REST 層 1 の ③）の材料は、1 つ下の段の要約である。
CHILD = {"年": "月", "月": "日"}
