"""黙っているあいだに届いたものを溜め、明けたときに 1 つの求めへ渡す（情-h・2026-09-16）。

沈黙の判定は以前、発話の**出口**（配信ゲート）にあった。そのため発話ごとに求めが立って
調停・調べもの・主LLM が回ってから止まり、他人への返事を通す例外（案イ）が入り込み、止めた
返事の配り時も限られた。**入口で止める**：黙っているあいだ、会話入力・機器・情動は O に
「聞いた／起きた／湧いた」として残すだけで求めを立てない。明けた瞬間に 1 つの求めを立て、
溜めたものを W の作業状態の枠に列挙し、主LLM が 1 回でまとめて答える。

ここは純関数：何を通すか（`lifts`）と、列挙の文（`render`）。器（`_muted`）はループが持つ。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .silence_rules import is_release

#: 本人が止めたいとき（`cancel_timer`・`/timer stop` に向かう言葉）は黙っていても通す。
_STOP = re.compile(r"止めて|ストップ|やめて")


@dataclass(frozen=True)
class Heard:
    """聞けなかったあいだに届いた 1 件（黙っていた・誰も見えなかった）。"""

    kind: str  # 会話入力／機器／情動
    text: str
    who: str = ""
    obs_id: str = ""
    at: float = 0.0  # epoch 秒
    why: str = "黙っていた"  # 聞けなかった理由：黙っていた／誰も見えなかった（2026-09-17）


def lifts(kind: str, text: str, *, speaker: str, asker: str) -> bool:
    """黙っていても求めを立ててよいきっかけか。

    - タイマーが鳴る（機器「タイマー」）：鳴ってほしいと決めた（知-n）。
    - 頼んだ本人の「話していい」：解く言葉そのもの。
    - 頼んだ本人の止める頼み（止めて・ストップ）：途中で停められることを軸にした（知-n）。
    """
    if kind == "機器" and text == "タイマー":
        return True
    if kind == "会話入力" and speaker and speaker == asker:
        return is_release(text) or bool(_STOP.search(text or ""))
    return False


def render(items: "list[Heard]", *, since: float, until: float, max_chars: int) -> str:
    """作業状態の枠に載せる列挙。会話は新しいものから原文、機器・情動は件数。

    上限を超えた古い会話は「ほか N 件（記憶にある）」の 1 行にまとめる（原文は O にある）。
    """
    if not items:
        return ""
    talks = [h for h in items if h.kind == "会話入力"]
    events = [h for h in items if h.kind == "機器"]
    urges = [h for h in items if h.kind == "情動"]
    seen = {h.why for h in items}
    whys = "／".join(w for w in ("黙っていた", "誰も見えなかった") if w in seen) or "聞けなかった"
    head = (
        f"{whys}あいだ（{_hm(since)}〜{_hm(until)}）に届いたもの——"
        f"聞いたこと {len(talks)} 件・起きたこと {len(events)} 件・湧いたこと {len(urges)} 件"
        "（全部を踏まえて、1 回でまとめて答える）："
    )
    tail: list[str] = []
    if events:
        tail.append("- 起きたこと：" + "・".join(h.text for h in events))
    if urges:
        counts: dict[str, int] = {}
        for h in urges:
            counts[h.text] = counts.get(h.text, 0) + 1
        tail.append("- 湧いたこと：" + "・".join(f"{k} ×{n}" for k, n in counts.items()))
    used = len(head) + sum(len(t) + 1 for t in tail)
    lines: list[str] = []
    left_out = 0
    for h in reversed(talks):  # 新しいものから
        line = f"- {_hm(h.at)} {h.who or '誰か'}：「{h.text}」"
        if lines and used + len(line) + 1 > max_chars:
            left_out += 1
            continue
        lines.append(line)
        used += len(line) + 1
    lines.reverse()  # 載せるぶんは時系列で
    if left_out:
        lines.insert(0, f"（ほか {left_out} 件は記憶にある）")
    return "\n".join([head, *lines, *tail])


def _hm(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch).astimezone().strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return "--:--"
