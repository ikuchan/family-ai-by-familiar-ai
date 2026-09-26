"""黙っているあいだに届いたものを溜め、明けたときに 1 つの求めへ渡す（情-h・2026-09-16）。

沈黙の判定は以前、発話の**出口**（配信ゲート）にあった。そのため発話ごとに求めが立って
調停・調べもの・主LLM が回ってから止まり、他人への返事を通す例外（案イ）が入り込み、止めた
返事の配り時も限られた。**入口で止める**：黙っているあいだ、会話入力・機器・情動は O に
「聞いた／起きた／湧いた」として残すだけで求めを立てない。明けた瞬間に 1 つの求めを立て、
溜めたものを W の作業状態の枠に列挙し、主LLM が 1 回でまとめて答える。

ここは純関数：何を通すか（`lifts`）と、列挙の文（`render`）。器（`_muted`）はループが持つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .silence_rules import is_release, names_me


@dataclass(frozen=True)
class Heard:
    """黙っていたあいだに届いた 1 件。"""

    kind: str  # 会話入力／機器／情動
    text: str
    who: str = ""
    obs_id: str = ""
    at: float = 0.0  # epoch 秒


def lifts(kind: str, text: str, *, names: "list[str]", reason: str = "") -> bool:
    """黙っていても求めを立ててよいきっかけか。

    - タイマーが鳴る（機器「タイマー」）：鳴ってほしいと決めた（知-n）。
    - **名前と「話していい」が同じ発話にある**：解く言葉。**誰の言葉でもよい**（出-as §2.5・2026-09-26）。
      以前は頼んだ本人の「話していい」と「止めて・ストップ」を通していたが、話者が分からない家では
      本人かを決められなかった。窓が開いているだけでは解かない（黙っているあいだ窓は開かない）。
    - **タイマー由来の沈黙**（`reason` が `timer:`）では、**名前つきの**タイマーの操作の言葉（「パジュ、止めて」・
      止め・一時停止・再開・12 字以内）は**誰の言葉でも**通す（情-m・2026-09-18。名前は出-au 段 1-2 で足した）。本人かは話者の指定（60 秒・知-t）で見るが、
      タイマー中は返事をしないので掛けて 60 秒後には本人が「分からない」になり、本人の「一時停止」が
      落ちた。声の門（`TIMER_MIC_CLOSE`）は既に「操作の言葉だけ・誰でも」なので入口も揃える。
    """
    from .timer_rules import is_control_word

    if kind == "機器" and text == "タイマー":
        return True
    if (
        kind == "会話入力"
        and reason.startswith("timer:")
        and is_control_word(text)
        and names
        and names_me(text, names)
    ):
        return True
    if kind == "会話入力" and is_release(text) and names and names_me(text, names):
        return True
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
    head = (
        f"黙っていたあいだ（{_hm(since)}〜{_hm(until)}）に届いたもの——"
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
