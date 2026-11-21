"""パジュの状態図を、Lean 4 の模型にして、決まりを証明する（環-bb・2026-11-21）。

本人の決定 (5)（2026-11-17）「Lean で確かめる決まりも、図と一緒に考える」と、2026-11-21 の承認「リーンは入れていいです」。
状態図の型（`PajuState`）から、領域ごとに状態の型と移りの関数を作り、次の決まりを `decide` で証明する（仮・本人の確認待ち）。

- **行き止まりが無い**：どの領域でも、どの状態からも、事象を送れば初めの状態へ戻れる。
- **寝ているあいだ耳は言葉を頭へ渡さない**（決定ク・ダ）：頭が寝ているとき `request_opened` を送っても、頭も耳の考えているあいだも
  移らない。

証明するのは、読み出した状態と移りの形だけ。移りの中の動作やデータの動きは、Python の試験が受け持つ。
使い方：`uv run python -m familiar_agent.loop.paju_state_lean > lean/PajuState.lean`、確かめるのは `lean lean/PajuState.lean`。
"""

from __future__ import annotations

from typing import Any

from .paju_state import PajuState
from .paju_state_chart import _walk


def _regions() -> "list[Any]":
    """状態を直に並べる領域（並行でない入れ物で、子に状態を持つもの）。考えているの子は内側の移りだけなので、頭の中で見る
    （外へ出る移りが足されたら `_moves` が誤りにする）。"""
    out = []
    for top in PajuState.states:
        for s in _walk(top):
            if s.states and not s.parallel and s.parent is not None and s.parent.parallel:
                out.append(s)
    return out


def _cap(ident: str) -> str:
    return "".join(p.capitalize() for p in ident.split("_"))


def _conds(t: Any) -> "list[tuple[str, bool]]":
    return [(sp.attr_name, bool(sp.expected_value)) for sp in t.cond.list if not sp.is_convention]


def _target(t: Any, region: Any) -> Any:
    target = t.target
    while target.parent is not region:
        if target.parent is None:
            raise ValueError(f"{t.event} の行き先が {region.id} の中に無い")
        target = target.parent
    return target


def _q(ident: str) -> str:
    """Lean の名前として包む（`end` などの予約語に当たっても読めるように）。"""
    return f"«{ident}»"


def _moves(region: Any) -> "list[tuple[Any, str, list[tuple[str, bool]], Any]]":
    """領域の中の移り (元の子, 事象, 条件, 行き先の子) を、ライブラリが試す順に返す。

    元は領域の直下の子。入れ子の中の状態から出る移りは、内側の移り（入れ子の中の子どうし）だけを許し、外へ出るものは誤りにする
    （模型が黙って落とさないように）。1 つの移りに事象が複数付いていれば、事象ごとに分ける。
    """
    out = []
    for kid in region.states:
        for s in _walk(kid):
            for t in s.transitions:
                if not t.event or t.internal:
                    continue
                if s is not kid:
                    raise ValueError(
                        f"{s.id} から出る {t.event} は入れ子の中から外へ出る（模型にできない）"
                    )
                for ev in t.event.split():
                    out.append((kid, ev, _conds(t), _target(t, region)))
    return out


def lean() -> str:
    regions = _regions()
    events: "list[str]" = []
    conds: "list[str]" = []
    for r in regions:
        for _kid, ev, cs, _tgt in _moves(r):
            if ev not in events:
                events.append(ev)
            for name, _ in cs:
                if name not in conds:
                    conds.append(name)
    out = [
        "/-! パジュの状態図の模型（`loop/paju_state_lean.py` がコードから作る。手で直さない）。 -/",
        "",
        "set_option linter.unusedVariables false",
        "",
        "inductive Ev where",
        *[f"  | {_q(e)}" for e in events],
        "  deriving DecidableEq, Repr",
        "",
        "def Ev.all : List Ev := [" + ", ".join(f".{_q(e)}" for e in events) + "]",
        "",
    ]
    params = " ".join(f"({_q(c)} : Bool)" for c in conds)
    for r in regions:
        name = _cap(r.id)
        kids = list(r.states)
        initial = next(k for k in kids if k.initial)
        out += [
            f"inductive {name} where",
            *[f"  | {_q(k.id)}" for k in kids],
            "  deriving DecidableEq, Repr",
            "",
            f"def {name}.all : List {name} := [" + ", ".join(f".{_q(k.id)}" for k in kids) + "]",
            "",
            " ".join(filter(None, [f"def {name}.step", params, f": {name} → Ev → Option {name}"])),
        ]
        # 同じ元と事象の移りは、ライブラリが書いた順に試し、条件の通った最初が走る。if の鎖にする
        grouped: "dict[tuple[str, str], list[tuple[list[tuple[str, bool]], Any]]]" = {}
        order: "list[tuple[str, str]]" = []
        for kid, ev, cs, tgt in _moves(r):
            key = (kid.id, ev)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append((cs, tgt))
        for src, ev in order:
            body = "none"
            for cs, tgt in reversed(grouped[(src, ev)]):
                guard = " && ".join(_q(c) if want else f"!{_q(c)}" for c, want in cs)
                body = (
                    f"if {guard} then some .{_q(tgt.id)} else {body}"
                    if guard
                    else f"some .{_q(tgt.id)}"
                )
            out.append(f"  | .{_q(src)}, .{_q(ev)} => {body}")
        out += ["  | _, _ => none", ""]
        falses = " ".join("false" for _ in conds)
        step = " ".join(filter(None, [f"{name}.step", falses]))
        out += [
            f"/-- {r.id}：n 歩以内に初めの状態（{initial.id}）へ戻れるか（条件がみな偽のときの移りで見る）。 -/",
            f"def {name}.back (n : Nat) (s : {name}) : Bool :=",
            "  match n with",
            f"  | 0 => s == .{_q(initial.id)}",
            f"  | n + 1 => s == .{_q(initial.id)} || Ev.all.any (fun e => match {step} s e with"
            f" | some t => {name}.back n t | none => false)",
            "",
            f"theorem {r.id}_has_no_dead_end : {name}.all.all ({name}.back {len(kids)}) = true := by decide",
            "",
        ]
    # 寝ているあいだ耳は言葉を頭へ渡さない（決定ク・ダ）。`is_sleeping` は頭が寝ているかそのもの（`PajuState.is_sleeping`）
    # なので、頭と耳のすべての組で、条件を頭の状態から作って確かめる。
    if {"head", "listening"} <= {r.id for r in regions} and "is_sleeping" in conds:

        def args(h: str) -> str:
            return " ".join(
                f"({h} == .{_q('sleeping')})" if c == "is_sleeping" else "false" for c in conds
            )

        out += [
            "/-- 頭が寝ているとき、`request_opened` では頭も耳の考えているあいだも移らない（決定ク・ダ）。 -/",
            "theorem sleeping_takes_no_request :",
            "    Head.all.all (fun h => Listening.all.all (fun l =>",
            f"      h != .{_q('sleeping')} ||",
            f"        (Head.step {args('h')} h .{_q('request_opened')} == none &&",
            f"         Listening.step {args('h')} l .{_q('request_opened')} == none))) = true := by",
            "  decide",
            "",
        ]
    return "\n".join(out)


if __name__ == "__main__":
    print(lean(), end="")
