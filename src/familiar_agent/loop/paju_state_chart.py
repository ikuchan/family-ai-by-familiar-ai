"""パジュの状態図を、コードから Mermaid の図にする（環-bb 一段目・2026-11-21）。

本人の決定 (2)（2026-11-17）「図はコードから自動で作る。読み出し、Mermaid の文字にし、見る、の 3 つの段で、図は手で書かない」。
状態図の型（`PajuState`）から、領域・状態・移り・事象を読み出す。設計の文書（`設計方針_パジュの状態` §2.1）は、ここが出す
文字をそのまま貼り、試験（`tests/test_paju_state_chart.py`）が一致を確かめる。

描かないもの：初めの状態へ入る移り（事象が無い）、内側の移り（考えているの子へ移るもの）、自分へ戻る移り（外へ出て入り直す）。
同じ 2 つの状態のあいだの事象は「、」でつなぐ。条件は事象のあとに（）で書く。入れ子の中の状態へ直に向かう
移りは、入れ子の箱へ向け、事象のあとに「（行き先へ）」と添える。

使い方：`uv run python -m familiar_agent.loop.paju_state_chart`

ライブラリの口（`statemachine.contrib.diagram.MermaidGraphMachine`）は使わない。名前が id から作った英語になり、入るときの動作の
行がどの状態にも付き、入れ子の状態へ入る移りを最初の子への移りに描き換え（「何もしていない → 思い出している」）、条件を描かない。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator

from .paju_state import _NAMES, PajuState

#: 領域の名前（`_NAMES` は状態の名前だけを持つ）。
_REGIONS = {
    "paju": "パジュ",
    "head": "頭",
    "ear": "耳",
    "listening": "考えているあいだ",
    "window": "窓",
    "awaited": "待っている答え",
    "entrance": "入口",
    "mouth": "口",
    "hush": "黙るか",
    "voice": "声",
    "alarm": "知らせの音",
}

#: 図の上で名前が重ならないように付けた名前（状態の行の名前とは別。同じ状態である）。
_DRAWN = {
    "no_answer": "答えなし",
    "no_voice": "声なし",
    "ear_open": "入口が開いている",
    "alarm_quiet": "音止まっている",
    "alarm_ringing": "音鳴っている",
}

#: 条件の読み方（条件の名前 → (真のときの書き方, 偽のときの書き方)）。
_CONDS = {"is_sleeping": ("頭が寝ていれば", "頭が寝ていなければ")}

_INDENT = "    "


def _name(state: Any) -> str:
    ident = state.id
    if ident in _REGIONS:
        return _REGIONS[ident]
    return _DRAWN.get(ident) or _NAMES[ident]


def _walk(state: Any) -> Iterator[Any]:
    yield state
    for child in getattr(state, "states", None) or []:
        yield from _walk(child)


def _drawn(transition: Any) -> bool:
    return bool(transition.event) and not transition.internal and not transition.is_self


def _label(transition: Any) -> str:
    """事象の名前と、付いていれば条件（例：`request_opened（頭が寝ていなければ）`）。"""
    notes = []
    for spec in transition.cond.list:
        if spec.is_convention:
            continue
        if spec.attr_name not in _CONDS:
            raise ValueError(f"条件 {spec.attr_name!r} の読み方を `_CONDS` に足す")
        when, unless = _CONDS[spec.attr_name]
        notes.append(when if spec.expected_value else unless)
    return transition.event + (f"（{'、'.join(notes)}）" if notes else "")


def _edges(compound: Any) -> "list[str]":
    """直下の子から出る移りを、(元, 先) ごとに事象をつないだ行にする。順は定義の順。"""
    order: "list[tuple[str, str]]" = []
    labels: "dict[tuple[str, str], list[str]]" = {}
    for child in compound.states:
        for t in child.transitions:
            if not _drawn(t):
                continue
            # 入れ子の中の状態へ直に向かう移りは、その入れ子の箱へ向け、事象のあとに行き先を添える。直に向けると、
            # Mermaid が行き先を箱の外に作る（2026-11-21 に描いて確かめた）。
            target = t.target
            while target.parent is not compound:
                if target.parent is None:
                    raise ValueError(
                        f"{t.event} の行き先が {compound.id} の中に無い（領域をまたぐ移りは描けない）"
                    )
                target = target.parent
            key = (_name(t.source), _name(target))
            if key not in labels:
                order.append(key)
                labels[key] = []
            label = _label(t) + (f"（{_name(t.target)}へ）" if target is not t.target else "")
            if label not in labels[key]:
                labels[key].append(label)
    return [f"{a} --> {b} : {'、'.join(labels[(a, b)])}" for a, b in order]


def _block(state: Any, depth: int) -> "list[str]":
    pad = _INDENT * depth
    lines = [f"{pad}state {_name(state)} {{"]
    inner = _INDENT * (depth + 1)
    children = list(state.states)
    if state.parallel:
        for i, child in enumerate(children):
            if i:
                lines.append(f"{inner}--")
            lines.extend(_block(child, depth + 1))
    else:
        # 入れ子の箱を先に出す（移りの行が先だと、箱の中の状態が外の領域にできることがある）
        for child in children:
            if child.states:
                lines.extend(_block(child, depth + 1))
        edges = _edges(state)
        ends = {n for e in edges for n in e.split(" : ")[0].split(" --> ")}
        for child in children:
            if not child.states and _name(child) not in ends:
                lines.append(inner + _name(child))  # 移りに出てこない状態も、箱の中に名前を出す
        lines.extend(inner + e for e in edges)
    lines.append(f"{pad}}}")
    return lines


def names() -> "list[str]":
    """図に出る状態と領域の名前（試験が、重ならないことを確かめる。Mermaid の名前は図の全体で 1 つ）。"""
    return [_name(s) for top in PajuState.states for s in _walk(top)]


def mermaid() -> str:
    """状態図の全体を Mermaid の文字にする（`stateDiagram-v2`・上から下）。末尾は改行で終わる。"""
    lines = ["stateDiagram-v2", f"{_INDENT}direction TB"]
    for top in PajuState.states:
        lines.extend(_block(top, 1))
    return "\n".join(lines) + "\n"


def events() -> "list[str]":
    """図に描く事象の名前（重なりなし・定義の順）。"""
    out: "list[str]" = []
    for top in PajuState.states:
        for state in _walk(top):
            for t in state.transitions:
                if _drawn(t) and t.event not in out:
                    out.append(t.event)
    return out


# ── いまの状態を光らせる（三段目）、ログから読む（四段目） ──────────────────────────

#: 光らせる色（Mermaid の `classDef`）。日本語の名前には `class 名前 now` が通らないので、`名前:::now` で付ける。
_NOW = "    classDef now fill:#ffd54f,stroke:#e65100,stroke-width:3px\n"


def active_names(sm: Any) -> "set[str]":
    """動いている状態図のいまの状態を、図の上の名前で返す（領域は入れない）。"""
    return {_name(s) for s in sm.configuration if s.id not in _REGIONS}


def highlighted(active: "Iterable[str]") -> str:
    """図に、`active` の状態を光らせる指定を足す。"""
    return mermaid() + _NOW + "".join(f"    {n}:::now\n" for n in sorted(active))


_MOVED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S* .*状態図 (\S+) で移った（(.*)）\s*$")


def transitions_in(lines: "Iterable[str]") -> "list[tuple[str, str, str]]":
    """ログの行から、移りの 1 行（`after_transition`）を (時刻, 事象, 各領域の状態) にして返す。ほかの行は捨てる。"""
    out = []
    for line in lines:
        m = _MOVED.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


if __name__ == "__main__":
    print(mermaid(), end="")
