"""拡散想起 (B) エンティティ辺の純ロジック（LLM フリー・機械的）。

現 W の MI の**関係の面**から、再想起の種にする person を選ぶ。話題の主体（`about`）を
最優先、次いでそばに居た人（`present`）、やった人（`actor`）。自分・現話者・既定話者は
除外する（役割1 の r で既に効く話者を二重に数えない）。想起本体は後続。

段4 で視点列（`writer_id`／`subject_id`／`participants_json`）から面へ移した。人と記憶の
結びつきは situated が担う（[D-在席相関/V2]）。役割名は 047 が定めたものである。
"""

from __future__ import annotations

from typing import Callable


def diffuse_ids(
    seed_ids: "list[str]",
    get_candidates: "Callable[[list[str]], list[str]]",
    *,
    max_add: int,
    max_depth: int,
) -> "list[str]":
    """有界再帰の spreading：seed から候補を辿り、現在集合に無い MI を最大 max_add 件足す。

    `get_candidates(known)`＝現在の既知集合から候補 id を返すコールバック（(A)+(B) を合成）。
    足した id を含めて再度候補を取り（想起が想起を呼ぶ）、深さ max_depth・件数 max_add で打ち切る。
    追加した順の id リストを返す（seed 自身・重複は除く）。DB も採点も持たない純関数。
    """
    known = list(dict.fromkeys(str(s) for s in seed_ids if s))
    known_set = set(known)
    added: list[str] = []
    for _ in range(max(0, max_depth)):
        if len(added) >= max_add:
            break
        fresh = [
            c
            for c in dict.fromkeys(str(x) for x in get_candidates(known) if x)
            if c not in known_set
        ]
        if not fresh:
            break
        for c in fresh[: max_add - len(added)]:
            known.append(c)
            known_set.add(c)
            added.append(c)
    return added


#: 種にする役割と、その優先順。`about`（話題の主体）→ `present`（そばに居た）→
#: `actor`（やった人）。047 が機械で立てるのは `actor` と `present` だけで、`about` は
#: REST 内省（記-a-ほ）が本文から抽出して足すまで空である。
SEED_ROLE_ORDER = ("about", "present", "actor")


def select_entity_seeds(
    relations: "list[dict]", exclude: "set[str] | frozenset[str] | None" = None
) -> "list[str]":
    """関係の面から再想起の種 person を選ぶ。`about`→`present`→`actor` の順で重複除去。

    受け取るのは面の行（`person_id`／`relation_key`）のリストである。
    """
    ex = set(exclude or ())
    seen: set[str] = set()
    out: list[str] = []

    def add(pid) -> None:
        p = str(pid) if pid else ""
        if p and p not in ex and p not in seen:
            seen.add(p)
            out.append(p)

    for role in SEED_ROLE_ORDER:
        for row in relations:
            if str(row.get("relation_key") or "") == role:
                add(row.get("person_id"))
    return out


def interleave_orders(
    far: "list[str]", stale: "list[str]", *, max_add: int, far_share: float
) -> "list[str]":
    """2 つの並び（遠い順・思い出していない順）から交互に取り、`max_add` 件にする（記-a-ろ-い）。

    `far_share` は遠い順から取る割合（0.5 なら 2／2）。同じ記録は 1 件と数え（重複は飛ばして
    次を取る）、片方が尽きたか枠を使い切ったら、もう片方で埋める。分類上遠いもの（新規性）と、
    長く思い出していないもの（掘り起こし）の両方を W に上げるため。
    """
    n = max(0, int(max_add))
    n_far = max(0, min(n, round(n * max(0.0, min(1.0, far_share)))))
    n_stale = n - n_far
    out: list[str] = []
    seen: set[str] = set()
    pos = {"far": 0, "stale": 0}
    lists = {"far": [str(x) for x in far], "stale": [str(x) for x in stale]}
    quota = {"far": n_far, "stale": n_stale}
    taken = {"far": 0, "stale": 0}

    def take(name: str) -> bool:
        """その並びから、まだ載っていない次の 1 件を取る。取れたら True。"""
        src = lists[name]
        while pos[name] < len(src):
            cand = src[pos[name]]
            pos[name] += 1
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
                taken[name] += 1
                return True
        return False

    turn = "far"
    while len(out) < n:
        order = (turn, "stale" if turn == "far" else "far")
        # 枠が残っている側を、いまの番から順に試す。
        picked = False
        for name in order:
            if taken[name] < quota[name] and take(name):
                picked = True
                break
        if not picked:
            # 枠は使い切った（か尽きた）。残りは取れる側で埋める。
            if not (take("far") or take("stale")):
                break
        turn = "stale" if turn == "far" else "far"
    return out[:n]
