"""経験繋がり（出-bq・2026-10-29・本人の決定）の純関数。DB は使わない。

REST が直近 7 日の日ごとの塊から、いちばん近い対がいちばん遠い 3 件を選び、種類「経験繋がり」の
1 つの関係にする。1 つの塊に 1 本で、よりよい 3 件が現れたら同じ関係の項を入れ替える。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from familiar_agent.core import experience_links as xl
from familiar_agent.core.experience_links import Action, Edge, day_clusters, pick, plan
from familiar_agent.store import relations

JST = timezone(timedelta(hours=9))


def _at(day: int, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 10, day, hh, mm, ss, tzinfo=JST)


def _exchange(origin: str, when: datetime) -> Edge:
    return Edge("やりとり", when, ((origin, "起点"), (f"{origin}-答え", "答え")))


def _co(ids, when: datetime) -> Edge:
    return Edge("共起", when, tuple((i, "項") for i in ids))


# ── 日ごとの塊（境界ア） ───────────────────────────────────────────────


def test_a_cluster_follows_only_the_edges_written_that_day():
    edges = [
        _exchange("s", _at(20, 23, 0)),
        _co(["s", "a", "b", "c", "d"], _at(20, 23, 59, 30)),
        _co(["s", "e", "f"], _at(21, 0, 0, 30)),  # 日本時間の翌日に書かれた辺はたどらない
    ]
    got = day_clusters(edges, {}, JST)
    assert got == {"2026-10-20": [frozenset("sabcd")]}


def test_the_same_edge_written_before_midnight_is_followed():
    """反証の側：0 時の前に書かれていれば、同じ辺をたどる。"""
    edges = [
        _exchange("s", _at(20, 23, 0)),
        _co(["s", "a", "b", "c", "d"], _at(20, 23, 59, 30)),
        _co(["s", "e", "f"], _at(20, 23, 59, 50)),
    ]
    got = day_clusters(edges, {}, JST)
    assert got == {"2026-10-20": [frozenset("sabcdef")]}


def test_a_cluster_without_an_origin_of_that_day_is_not_counted():
    edges = [_co(["x", "y", "z", "w", "v"], _at(20, 12, 0))]
    assert day_clusters(edges, {}, JST) == {}


def test_a_final_version_is_joined_to_its_origin():
    """共起の端は起点、継起の端は最終版なので、親子（最終版の parent_id＝起点）でつなぐ。"""
    edges = [
        _exchange("s", _at(20, 10, 0)),
        Edge("継起", _at(20, 10, 5), (("f0", "前"), ("f1", "後"))),
    ]
    assert day_clusters(edges, {"f1": "s"}, JST) == {"2026-10-20": [frozenset({"s", "f0", "f1"})]}
    # 反証の側：親が分からなければ、起点は網に入らない
    assert day_clusters(edges, {}, JST) == {}


# ── 3 件の選び出し（距離ア） ─────────────────────────────────────────────


def _unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float32)


def test_pick_returns_the_triple_whose_closest_pair_is_farthest():
    """いちばん遠い対から始めると (0°,170°,240°) になる配置で、答えは (0°,120°,240°)。"""
    vecs = {"p0": _unit(0), "p1": _unit(170), "p2": _unit(120), "p3": _unit(240)}
    triple, sim = pick(list(vecs), vecs, exact_max=100)
    assert set(triple) == {"p0", "p2", "p3"}
    assert math.isclose(sim, -0.5, abs_tol=1e-5)


def test_pick_falls_back_to_the_greedy_choice_above_the_limit():
    """反証の側：上限を超えると、いちばん遠い対から 3 件目を足す近似になり、答えが変わる。"""
    vecs = {"p0": _unit(0), "p1": _unit(170), "p2": _unit(120), "p3": _unit(240)}
    triple, sim = pick(list(vecs), vecs, exact_max=3)
    assert set(triple) == {"p0", "p1", "p3"}
    assert math.isclose(sim, math.cos(math.radians(70)), abs_tol=1e-5)


def test_a_zero_vector_counts_as_no_vector():
    """埋め込みの失敗で入る零の埋め込みは、意味を持たないので、無いものと見る（類似度が NaN になる）。"""
    vecs = {"a": _unit(0), "b": _unit(120), "c": _unit(240), "z": np.zeros(2, dtype=np.float32)}
    triple, _sim = pick(list(vecs), vecs, exact_max=100)
    assert set(triple) == {"a", "b", "c"}
    triple, _sim = pick(list(vecs), vecs, exact_max=2)  # 近似の側も
    assert set(triple) == {"a", "b", "c"}
    assert xl.closest_pair(["a", "b", "z"], vecs) == float("inf")
    assert math.isclose(xl.closest_pair(["a", "b", "c"], vecs), -0.5, abs_tol=1e-5)  # 反証の側


def test_the_words_match_the_relation_store():
    """関係の種類と役割の語は、関係の層の定数と同じ（ずれると塊が 1 つもできず、黙って何も書かない）。"""
    assert xl.TRAVERSED_KINDS == (relations.KIND_COOCCURRENCE, relations.KIND_SUCCESSION)
    assert xl.KIND_EXCHANGE == relations.KIND_EXCHANGE
    assert xl.ROLE_ORIGIN == "起点"


def test_pick_needs_three_records_with_vectors():
    vecs = {"a": _unit(0), "b": _unit(90)}
    assert pick(["a", "b", "c"], vecs, exact_max=100) is None


# ── 書く・入れ替える・何もしない ─────────────────────────────────────────


def _equiangular(c: float) -> "list[np.ndarray]":
    """どの 2 つの cos 類似度も c になる 3 本（4 次元）。"""
    k = c / (1 - c)
    t = (-2 + math.sqrt(4 + 12 * k)) / 6
    out = []
    for i in range(3):
        v = np.full(4, t, dtype=np.float64)
        v[3] = 0.0
        v[i] += 1.0
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return out


def _cluster_with(c: float):
    """どの 2 つも類似度 c の a・b・c と、a にほぼ重なる d・e の 5 件。選ばれるのは a か d か e と、b・c。"""
    a, b, cc = _equiangular(c)

    def near(v, eps):
        w = v.copy()
        w[3] = eps
        return (w / np.linalg.norm(w)).astype(np.float32)

    vecs = {"a": a, "b": b, "c": cc, "d": near(a, 0.05), "e": near(a, 0.1)}
    return frozenset(vecs), vecs


def _plan(clusters, links, vecs):
    return plan(clusters, links, vecs, gate=0.6, min_size=5, exact_max=100)


def _on(day: str, ids) -> xl.Link:
    return xl.Link(day, frozenset(ids))


D20, D21 = "2026-10-20", "2026-10-21"


def test_a_new_link_is_written_at_or_below_the_gate():
    cl, vecs = _cluster_with(0.59)
    got = _plan({"2026-10-20": [cl]}, {}, vecs)
    assert len(got) == 1 and got[0].op == "add" and got[0].relation_id is None
    assert got[0].day == "2026-10-20"  # 持ち主の日を添えて書く
    assert {"b", "c"} <= set(got[0].ids) and len(got[0].ids) == 3


def test_nothing_is_written_above_the_gate():
    cl, vecs = _cluster_with(0.61)
    assert _plan({"2026-10-20": [cl]}, {}, vecs) == []


def test_a_cluster_of_four_is_skipped_but_five_is_not():
    cl, vecs = _cluster_with(0.3)
    four = frozenset(["a", "b", "c", "d"])
    assert _plan({"2026-10-20": [four]}, {}, vecs) == []
    assert len(_plan({"2026-10-20": [cl]}, {}, vecs)) == 1


def _apply(actions, links):
    """関係の層がすることを、辞書の上でまねる。"""
    out = dict(links)
    nxt = max(out, default=0) + 1
    for a in actions:
        if a.op == "add":
            out[nxt] = _on(a.day, a.ids)
            nxt += 1
        elif a.op == "replace":
            out[a.relation_id] = _on(out[a.relation_id].day, a.ids)  # 時刻（持ち主の日）は保つ
        elif a.op == "remove":
            del out[a.relation_id]
    return out


def test_running_twice_keeps_one_link_per_cluster():
    """同じつながりを 2 つ作らない：2 回目は何もしない。"""
    cl, vecs = _cluster_with(0.3)
    clusters = {"2026-10-20": [cl]}
    links = _apply(_plan(clusters, {}, vecs), {})
    assert len(links) == 1
    assert _plan(clusters, links, vecs) == []


def test_a_better_triple_replaces_the_existing_link_in_place():
    cl, vecs = _cluster_with(0.3)
    worse = frozenset({"a", "d", "b"})  # a と d はほぼ重なる（0.99）
    got = _plan({"2026-10-20": [cl]}, {7: _on(D20, worse)}, vecs)
    assert [(a.op, a.relation_id) for a in got] == [("replace", 7)]
    assert {"b", "c"} <= set(got[0].ids)


def test_an_existing_link_that_is_as_good_is_kept():
    """入れ替えは厳密に低いときだけ。同じ組は何もしない。"""
    cl, vecs = _cluster_with(0.3)
    first = _plan({"2026-10-20": [cl]}, {}, vecs)[0]
    assert _plan({"2026-10-20": [cl]}, {3: _on(D20, first.ids)}, vecs) == []


def test_two_links_in_one_cluster_become_one():
    """塊が育って 2 本を含んだら、よいほうへ寄せ、残りは外す。"""
    cl, vecs = _cluster_with(0.3)
    first = frozenset(_plan({"2026-10-20": [cl]}, {}, vecs)[0].ids)
    got = _plan({"2026-10-20": [cl]}, {4: _on(D20, first), 9: _on(D20, "ade")}, vecs)
    assert got == [Action("remove", 9, ())]


def test_a_link_shared_by_two_days_does_not_churn():
    """古い記憶 3 件が 2 日の塊に入っても、2 回目以降は何も動かない（先の日が持ち主）。"""
    cl, vecs = _cluster_with(0.3)
    vecs = dict(vecs)
    vecs["g"] = _unit(45).tolist() + [0.0, 0.0]
    vecs["g"] = np.array(vecs["g"], dtype=np.float32)
    day2 = frozenset(cl | {"g"})
    clusters = {"2026-10-20": [cl], "2026-10-21": [day2]}
    links = _apply(_plan(clusters, {}, vecs), {})
    again = _apply(_plan(clusters, links, vecs), links)
    assert again == links
    assert _plan(clusters, links, vecs) == []


def test_a_day_that_left_the_window_keeps_its_link():
    """7 日の窓から出た日の経験繋がりは、同じ記憶を含む後の日の塊でも、入れ替えも外しもしない。"""
    cl, vecs = _cluster_with(0.3)
    worse = _on(D20, "adb")
    got = _plan({D21: [cl]}, {7: worse}, vecs)
    assert all(a.relation_id != 7 for a in got)
    # 反証の側：同じ日の関係なら入れ替える
    assert [(a.op, a.relation_id) for a in _plan({D21: [cl]}, {7: _on(D21, "adb")}, vecs)] == [
        ("replace", 7)
    ]
