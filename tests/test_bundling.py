"""核の束ね（`core/bundling.py`・記-a-ろ-に・`出来事を畳む` §3c）。純関数・numpy だけ。

- 同一：コサイン ≥ τ同 でつながる群を**個数上限なし**で。
- 覆い：半径 τ の球で全記録を覆うのに要る種の数 K_τ → 1 束の容量 k=⌈N/K_τ⌉。
- 束ね：farthest-first の種・容量 k の割り当て・中心の更新。中心とのコサイン < τ は孤立。
"""

from __future__ import annotations

import numpy as np

from familiar_agent.core import bundling as bd


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cloud(center, n, spread, rng):
    """center の周りに n 本（コサイン ≈ 1−spread）。"""
    out = []
    for _ in range(n):
        out.append(_unit(center + rng.normal(0.0, spread, size=len(center))))
    return out


def _two_hills(rng, n_a=8, n_b=8, dim=16):
    a = _unit(np.eye(dim)[0])
    b = _unit(np.eye(dim)[1])
    return np.stack(_cloud(a, n_a, 0.08, rng) + _cloud(b, n_b, 0.08, rng)), a, b


def test_duplicate_groups_join_only_the_near_identical_without_a_size_cap():
    rng = np.random.default_rng(0)
    base = _unit(np.eye(8)[0])
    same = [base] * 14  # 完全一致 14 本
    near = [_unit(base + rng.normal(0.0, 0.002, 8)) for _ in range(2)]  # ≈0.999
    other = [_unit(np.eye(8)[1]), _unit(np.eye(8)[2])]
    V = np.stack(same + near + other)
    groups = bd.duplicate_groups(V, 0.98)
    assert len(groups) == 1 and sorted(groups[0]) == list(range(16))
    assert bd.duplicate_groups(V[-2:], 0.98) == []


def test_cover_count_sees_the_hills():
    rng = np.random.default_rng(1)
    V, _a, _b = _two_hills(rng)
    assert bd.cover_count(V, 0.5) == 2
    assert bd.cover_count(V, 0.999) >= 8  # ほぼ完全一致しか同じ球に入らない
    assert bd.cover_count(V[:1], 0.5) == 1


def test_bundles_respect_the_capacity_and_mark_isolates():
    rng = np.random.default_rng(2)
    V, _a, _b = _two_hills(rng)
    lone = _unit(np.eye(16)[5])
    V = np.vstack([V, lone[None, :]])
    res = bd.bundle(V, tau=0.5, k_max=12, min_size=3)
    assert res.k == -(-17 // res.cover)  # k=⌈N/K⌉
    sizes = [len(b) for b in res.bundles]
    assert all(s <= res.k for s in sizes) and all(s >= 3 for s in sizes)
    assert all(16 not in b for b in res.bundles)  # 独りの記録はどの束にも入らない（m 未満で残る）
    assert sum(sizes) + len(res.isolated) + res.dropped == 17
    # 同じ丘は同じ束（丘 a の 8 本は容量 k で 1〜2 束に割れるが、丘をまたがない）
    for b in res.bundles:
        assert all(i < 8 for i in b) or all(8 <= i < 16 for i in b)


def test_k_max_caps_the_capacity_and_small_bundles_are_dropped():
    rng = np.random.default_rng(3)
    V, _a, _b = _two_hills(rng, n_a=20, n_b=2)
    res = bd.bundle(V, tau=0.5, k_max=6, min_size=3)
    assert res.k <= 6
    assert all(len(b) <= 6 for b in res.bundles)
    assert res.dropped == 2  # 丘 b の 2 本は m=3 未満の束なので落ちる（残る）


def test_bundling_is_deterministic():
    rng = np.random.default_rng(4)
    V, _a, _b = _two_hills(rng)
    r1 = bd.bundle(V, tau=0.5, k_max=12, min_size=3)
    r2 = bd.bundle(V, tau=0.5, k_max=12, min_size=3)
    assert r1.bundles == r2.bundles and r1.isolated == r2.isolated


def test_center_similarity_is_reported_per_bundle():
    rng = np.random.default_rng(5)
    V, _a, _b = _two_hills(rng)
    res = bd.bundle(V, tau=0.5, k_max=12, min_size=3)
    assert len(res.center_sim) == len(res.bundles)
    assert all(0.5 <= s <= 1.0 for s in res.center_sim)


def test_empty_input_is_fine():
    res = bd.bundle(np.zeros((0, 4), dtype=np.float32), tau=0.5, k_max=12, min_size=3)
    assert res.bundles == [] and res.isolated == [] and res.k == 0 and res.cover == 0
    assert bd.duplicate_groups(np.zeros((0, 4), dtype=np.float32), 0.98) == []
