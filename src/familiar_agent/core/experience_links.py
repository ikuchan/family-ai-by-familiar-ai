"""経験繋がり（出-bq・2026-10-29・本人の決定）の純関数。DB を触らない。

会話の流れで生まれた**意味の似ていない飛び**を、REST が日ごとにまとめて関係にする（`課題8` 出-bq）。

1. **日ごとの塊**（境界ア）：その日に始まったやりとりの起点から、その日に書かれた共起と継起の辺だけを
   たどる。共起の端は起点、継起の端は最終版なので、最終版は親（`parent_id`＝起点）とつなぐ。
2. **3 件の選び出し**（距離ア）：どの 2 つの間も遠くなる 3 件（いちばん近い対がいちばん遠い組）を選ぶ。
3. **書くかどうか**：いちばん近い対の類似度が門以下なら書く。値は残さない（門にだけ使う）。
   1 つの塊に 1 本で、塊の中に既存の経験繋がり（同じ日のもので、3 件ともその塊に入っているもの）があれば、
   新しい 3 件のほうが**厳密に**近い対の類似度が低いときだけ、同じ関係の項を入れ替える。
   持ち主の日は関係の時刻に持つ（7 日の窓から出た日の関係に、後の日の塊が触らない）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, tzinfo

import numpy as np

#: 塊をたどる辺。やりとりは起点を見つけるのに使い、辺としてはたどらない。
#: 語は関係の層（`store/relations.py`）の定数と同じ。純関数なので層を読まず、試験で一致を守る。
TRAVERSED_KINDS = ("共起", "継起")
KIND_EXCHANGE = "やりとり"
ROLE_ORIGIN = "起点"


@dataclass(frozen=True)
class Edge:
    """関係 1 本。`members` は `(記録の id, 役割)` の並び。"""

    kind: str
    created_at: datetime
    members: "tuple[tuple[str, str], ...]"


@dataclass(frozen=True)
class Link:
    """既存の経験繋がり 1 本。`day` は持ち主の塊の日（関係の時刻の暦日）。"""

    day: str
    ids: "frozenset[str]"


@dataclass(frozen=True)
class Action:
    """関係の層にさせること。`op` は `add`（新しく書く）・`replace`（項を入れ替える）・`remove`（外す）。

    `day` は `add` のときの持ち主の塊の日（関係の時刻にする）。
    """

    op: str
    relation_id: "int | None"
    ids: "tuple[str, ...]"
    day: str = ""


def day_clusters(
    edges: "list[Edge]", parents: "dict[str, str]", tz: tzinfo
) -> "dict[str, list[frozenset[str]]]":
    """日（`tz` の暦日）ごとに、その日の起点を含む塊を返す。

    `parents` は最終版 → 起点。網に入っている最終版だけを親とつなぐ。塊は大きさを問わずに返す
    （5 件未満を飛ばすのは `plan`）。並びは決まった順（件数の多い順・同数なら id の順）。
    """
    by_day: "dict[str, list[Edge]]" = {}
    for e in edges:
        by_day.setdefault(e.created_at.astimezone(tz).date().isoformat(), []).append(e)
    out: "dict[str, list[frozenset[str]]]" = {}
    for day, day_edges in sorted(by_day.items()):
        adj: "dict[str, set[str]]" = {}
        starts: "set[str]" = set()
        for e in day_edges:
            ids = [o for o, _ in e.members]
            if e.kind == KIND_EXCHANGE:
                starts |= {o for o, role in e.members if role == ROLE_ORIGIN}
            elif e.kind in TRAVERSED_KINDS:
                for o in ids:
                    adj.setdefault(o, set())
                for a, b in itertools.combinations(ids, 2):
                    adj[a].add(b)
                    adj[b].add(a)
        for child, parent in parents.items():
            if child in adj:
                adj[child].add(parent)
                adj.setdefault(parent, set()).add(child)
        seen: "set[str]" = set()
        comps: "list[frozenset[str]]" = []
        for s in sorted(starts):
            if s not in adj or s in seen:
                continue
            comp: "set[str]" = set()
            stack = [s]
            while stack:
                x = stack.pop()
                if x not in comp:
                    comp.add(x)
                    stack.extend(adj[x] - comp)
            seen |= comp
            comps.append(frozenset(comp))
        if comps:
            out[day] = sorted(comps, key=lambda c: (-len(c), sorted(c)))
    return out


def _usable(i: str, vecs: "dict[str, np.ndarray]") -> bool:
    """埋め込みがあり、零でない（埋め込みの失敗で入る零は意味を持たず、類似度が NaN になる）。"""
    return i in vecs and float(np.linalg.norm(vecs[i])) > 1e-8


def _sims(ids: "list[str]", vecs: "dict[str, np.ndarray]") -> np.ndarray:
    m = np.array([vecs[i] for i in ids], dtype=np.float64)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m @ m.T


def closest_pair(ids, vecs: "dict[str, np.ndarray]") -> float:
    """その組のいちばん近い対の cos 類似度。埋め込みの無い記録があれば無限大（比べると必ず負ける）。"""
    ids = list(ids)
    if not all(_usable(i, vecs) for i in ids):
        return float("inf")
    s = _sims(ids, vecs)
    return float(max(s[a, b] for a, b in itertools.combinations(range(len(ids)), 2)))


def pick(
    ids, vecs: "dict[str, np.ndarray]", *, exact_max: int
) -> "tuple[tuple[str, str, str], float] | None":
    """いちばん近い対がいちばん遠い 3 件と、その対の類似度を返す。埋め込みのある記録が 3 件未満なら None。

    零の埋め込みは無いものと見る。`exact_max` 件まではすべての組を調べる。超えたら、いちばん遠い対から始め、その 2 つの
    どちらにも遠い 3 件目を足す近似にする（記-s の前のような数百件の塊で REST を止めない）。
    同点は id の順で先のものを取る（何度回しても同じ組になる）。
    """
    have = sorted(i for i in set(ids) if _usable(i, vecs))
    n = len(have)
    if n < 3:
        return None
    s = _sims(have, vecs)
    if n <= exact_max:
        best, best_sim = None, float("inf")
        for a, b, c in itertools.combinations(range(n), 3):
            v = max(s[a, b], s[a, c], s[b, c])
            if v < best_sim:
                best, best_sim = (a, b, c), v
        assert best is not None
        a, b, c = best
    else:
        masked = s + np.eye(n) * 9.0
        a, b = (int(x) for x in np.unravel_index(int(np.argmin(masked)), masked.shape))
        far = np.maximum(s[a], s[b])
        far[[a, b]] = np.inf
        c = int(np.argmin(far))
    triple = tuple(sorted((have[a], have[b], have[c])))
    return (triple[0], triple[1], triple[2]), float(max(s[a, b], s[a, c], s[b, c]))


def plan(
    clusters: "dict[str, list[frozenset[str]]]",
    links: "dict[int, Link]",
    vecs: "dict[str, np.ndarray]",
    *,
    gate: float,
    min_size: int,
    exact_max: int,
) -> "list[Action]":
    """塊と既存の経験繋がりから、関係の層にさせることを決める。

    日は古い順に見る。既存の経験繋がりは、同じ日の塊のうち 3 件とも入っている塊が持ち主になる
    （同じ日の塊は重ならない）。窓に無い日の関係は、どの塊も持たないので触らない。後の日は、
    先の日の塊にまるごと入る 3 件を書かない（古い記憶 3 件が 2 日の塊に入っても、取り合わない）。
    3 件がまったく同じ経験繋がりは、どこにあっても 2 本目を書かない。
    """
    actions: "list[Action]" = []
    taken: "set[frozenset[str]]" = {lk.ids for lk in links.values()}
    earlier: "list[frozenset[str]]" = []
    for day in sorted(clusters):
        for cl in clusters[day]:
            if len(cl) < min_size:
                continue
            mine = sorted(rid for rid, lk in links.items() if lk.day == day and lk.ids <= cl)
            got = pick(cl, vecs, exact_max=exact_max)
            inside_earlier = got is not None and any(set(got[0]) <= c for c in earlier)
            earlier.append(cl)
            if got is None:
                continue
            triple, sim = got
            if not mine:
                if sim <= gate and not inside_earlier and frozenset(triple) not in taken:
                    actions.append(Action("add", None, triple, day))
                    taken.add(frozenset(triple))
                continue
            keep = min(mine, key=lambda rid: (closest_pair(links[rid].ids, vecs), rid))
            for rid in mine:
                if rid != keep:
                    actions.append(Action("remove", rid, ()))
            better = sim < closest_pair(links[keep].ids, vecs) and sim <= gate
            if better and not inside_earlier and frozenset(triple) not in taken:
                actions.append(Action("replace", keep, triple))
                taken.discard(links[keep].ids)
                taken.add(frozenset(triple))
    return actions
