"""BOND の発火で使う想起の手がかり（情-q・2026-10-10・本人の決定）。純関数。

BOND の発火の W は、内的な促しの文（「誰かと居たい気持ちが湧いている…」）だけを手がかりに引いていたので、誰が居るか・
これから何があるかと結びつきにくかった。手がかりに、先読みした 7 日分の家族の予定（`loop/schedule_watch`）と、パジュへの
メモで最後に新しく書かれたこと（`loop/notes_watch`）を並べ、W を家族と予定に寄せる。手がかりは何に似た記憶を引くかを
決めるだけで、文そのものは W に載らない（予定とメモは O の記録として上がってくる）。
"""

from __future__ import annotations

#: 予定の部分の長さ（字）。7 日分は 1 行 20〜40 字で 15〜30 行ほど〔仮〕。
SCHEDULE_MAX = 600
#: メモの新しい行の長さ（字）。
MEMO_MAX = 300


def cue_for(urge: str, schedule: str, memo_added: "list[str]") -> str:
    """内的な促しの文に、予定（`SCHEDULE_MAX` 字まで）とメモの新しい行（`MEMO_MAX` 字まで）を並べる。無い部分は省く。"""
    parts = [urge]
    if schedule.strip():
        parts.append("家族のこれから 7 日の予定：\n" + schedule.strip()[:SCHEDULE_MAX])
    memo = "\n".join(ln for ln in memo_added if ln.strip()).strip()
    if memo:
        parts.append("パジュへのメモで新しく書かれたこと：\n" + memo[:MEMO_MAX])
    return "\n\n".join(parts)
