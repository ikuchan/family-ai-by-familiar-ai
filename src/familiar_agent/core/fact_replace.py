"""事実の置き換えを判定する頼みと、その返りの読み方（記-ah 段 2・2026-11-25）。純関数だけ。

候補の組は、同じ話題（事実の鍵の「答えを隠す向き」の文。例「さぶろうの習い事」）について、古い記録と新しい記録の 2 件で
ある。組ごとに両方の本文と日付を見せ、新しい記録の事実が古い記録の事実を置き換えたか（同時には成り立たないか）と、
置き換えた事実がいつから成り立ったか（文の中の「先月から」を日付に直したもの）を返させる。軽量LLM を呼ぶのと書くのは
`loop/rest_replace.py`。考え方は Zep の無効化（`設計方針_事実の移り変わり` §1）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from .helpers import strip_code_fence

MAX_CHARS = 400  # 1 件の本文を頼みに載せる上限（話題の前後が入れば足りる）
_ARRAY_START = re.compile(r"\[\s*[\{\]]")  # JSON の配列の始まり（`[{` か `[]`）
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PROMPT = """次の組それぞれについて、同じ話題について書かれた「古い記録」と「新しい記録」を比べてください。

- 新しい記録の事実が、古い記録の事実を置き換えた（話題の答えが変わり、両方が同時には成り立たない）なら replaced を true にする。
  例：古い「習い事はサッカー」、新しい「先月から水泳に変わった」は true。
- 両方が同時に成り立つ（付け足し、言い直し、別の時のこと、同じことの繰り返し）なら false にする。迷ったら false。
- true のとき、新しい事実がいつから成り立ったかが記録から分かれば、新しい記録の日付をもとに日付（YYYY-MM-DD）に直して
  since に書く（例：新しい記録が 2026-11-20 で「先月から」なら 2026-10-01）。分からなければ since は書かない。

出力は JSON だけにする（説明は書かない）：
[{"i": 組の番号, "replaced": true か false, "since": "YYYY-MM-DD"}]

組：
"""


@dataclass(frozen=True)
class Pair:
    """置き換えの候補の組。`topic` は話題の鍵の文、`*_key` はその鍵の文の行の番号。"""

    topic: str
    old_obs: str
    old_key: int
    old_text: str
    old_at: datetime
    new_obs: str
    new_key: int
    new_text: str
    new_at: datetime


@dataclass(frozen=True)
class Verdict:
    """組 1 つの判定。`since` は置き換えた事実がいつから成り立ったか（YYYY-MM-DD・分からなければ None）。"""

    replaced: bool
    since: "str | None" = None


def _line(text: str) -> str:
    return text[:MAX_CHARS].replace("\n", " ")


def build_prompt(pairs: "list[Pair]") -> str:
    """組を番号つきで並べた頼み。日付は家の暦ではなく記録の日付の欄（`%Y-%m-%d`）。"""
    rows = []
    for i, p in enumerate(pairs):
        rows.append(
            f"[{i}] 話題：{p.topic}\n"
            f"  古い記録（{p.old_at:%Y-%m-%d}）：{_line(p.old_text)}\n"
            f"  新しい記録（{p.new_at:%Y-%m-%d}）：{_line(p.new_text)}"
        )
    return PROMPT + "\n".join(rows)


def parse(raw: str, n: int) -> "list[Verdict | None] | None":
    """返りを組ごとの判定にする。読めなければ None。番号の合わない行と、判定の無い組は None。"""
    text = strip_code_fence(str(raw or ""))
    m = _ARRAY_START.search(text)
    if m is None:
        return None
    try:
        got = json.loads(text[m.start() : text.rfind("]") + 1])
    except ValueError:
        return None
    if not isinstance(got, list):
        return None
    out: "list[Verdict | None]" = [None] * n
    for row in got:
        if not isinstance(row, dict):
            continue
        i = row.get("i")
        if not isinstance(i, int) or not 0 <= i < n or not isinstance(row.get("replaced"), bool):
            continue
        since = row.get("since")
        ok = isinstance(since, str) and bool(_DATE.match(since))
        out[i] = Verdict(replaced=row["replaced"], since=since if ok else None)
    return out


def note(reps: list, tz) -> str:
    """置き換えられた事実の注記（記-ah 段 3）。「（「話題」は いつ に置き換わった・新しい記録 id:…）」を話題ごとに 1 つ。

    同じ古い事実が写しの記録ぶん置き換えられていても（段 2 は同じ本文の記録すべてに書く）、話題（古い鍵）ごとに 1 つにし、
    いちばん早い置き換えを書く。日付は `tz`（家の暦）で書く。W の行と、思い出す道具の返りの両方が使う。
    """
    first: dict = {}
    for rep in reps:
        cur = first.get(rep.old_key)
        if cur is None or (rep.valid_at and (cur.valid_at is None or rep.valid_at < cur.valid_at)):
            first[rep.old_key] = rep
    out = []
    for rep in first.values():
        when = rep.valid_at.astimezone(tz).strftime("%Y-%m-%d") if rep.valid_at else "?"
        sid = rep.new_obs.replace("-", "")[:12] or "?"
        out.append(f"（「{rep.old_sentence}」は {when} に置き換わった・新しい記録 id:{sid}）")
    return "".join(out)
