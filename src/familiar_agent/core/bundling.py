"""核の束ね（記-a-ろ-に・2026-09-15・`設計方針_REST内省_出来事を畳む` §3c）。純関数・numpy だけ。

②核の固めは「対象と束は機械が決め、LLM は 1 束 1 文を書くだけ」。ここはその機械の側。

- **同一**（`duplicate_groups`）：コサイン ≥ τ同 でつながる群。**個数上限なし**——情報が増えて
  いないものを容量で割って同じ要約を何本も作らないため。
- **覆い**（`cover_count`）：半径 τ の球で全記録を覆うのに要る種の数 K_τ。1 束の容量は
  k=⌈N/K_τ⌉ で、「この近さで分けると自然に何束になるか」から導く（k は自由な仮値にしない）。
- **束ね**（`bundle`）：farthest-first の種・容量 k の割り当て（似ている組から順に埋め、満ちた束は
  閉じる）・中心の更新を数回。最終の中心とのコサイン < τ は孤立（束に入れない）、m 件未満の束は
  落とす（残す）。決定的（乱数を使わない）。

本番の核（2026-09-15・107 件）の総当たりコサインは二山（p50 0.09・p90 0.69）で、τ=0.5 の覆いは
K=29・k=4、束の中身は意味が揃っていた。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit_rows(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float32)
    if V.ndim != 2 or V.shape[0] == 0:
        return np.zeros((0, V.shape[1] if V.ndim == 2 else 0), dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return V / norms


def duplicate_groups(V: np.ndarray, tau_same: float) -> list[list[int]]:
    """コサイン ≥ τ同 でつながる群（2 件以上）を、添字の並びで返す。"""
    U = _unit_rows(V)
    n = U.shape[0]
    if n < 2:
        return []
    C = U @ U.T
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(np.triu(C, 1) >= tau_same)):
        parent[find(int(i))] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _seeds(U: np.ndarray, tau: float) -> list[int]:
    """farthest-first。全記録がどれかの種から τ 以内に入るまで種を足す。"""
    n = U.shape[0]
    if n == 0:
        return []
    center = U.mean(axis=0)
    seeds = [int(np.argmax(np.linalg.norm(U - center, axis=1)))]
    while True:
        best = (U @ U[seeds].T).max(axis=1)
        if bool((best >= tau).all()):
            return seeds
        seeds.append(int(np.argmin(best)))


def cover_count(V: np.ndarray, tau: float) -> int:
    return len(_seeds(_unit_rows(V), tau))


@dataclass(frozen=True)
class Bundles:
    bundles: list[list[int]]  # 各束の添字（m 件以上・容量 k 以下）
    center_sim: list[float]  # 各束の中心コサインの平均
    isolated: list[int]  # 中心から τ 未満で束に入らなかった添字
    dropped: int  # m 件未満で落とした束に居た記録の数
    cover: int  # K_τ
    k: int  # 容量 ⌈N/K_τ⌉（k_max で切った後）
    seeds: list[int] = field(default_factory=list)


def _split_with_capacity(U: np.ndarray, members: list[int], k: int, rounds: int) -> list[list[int]]:
    """1 つの自然な群を容量 k の束に割る（群の中だけで farthest-first の種・容量つき割り当て）。"""
    if len(members) <= k:
        return [list(members)]
    M = U[members]
    parts = -(-len(members) // k)
    cap_each = -(-len(members) // parts)  # 大きさを揃える（例：8 件・k=6 → 4+4）
    center = M.mean(axis=0)
    seeds = [int(np.argmax(np.linalg.norm(M - center, axis=1)))]
    while len(seeds) < parts:
        best = (M @ M[seeds].T).max(axis=1)
        best[seeds] = np.inf
        seeds.append(int(np.argmin(best)))
    cent = M[seeds].copy()
    assign = -np.ones(len(members), dtype=int)
    for _ in range(max(1, int(rounds))):
        sim = M @ cent.T
        order = np.argsort(-sim, axis=None, kind="stable")
        assign = -np.ones(len(members), dtype=int)
        cap = np.zeros(parts, dtype=int)
        for flat in order:
            i, c = divmod(int(flat), parts)
            if assign[i] < 0 and cap[c] < cap_each:
                assign[i] = c
                cap[c] += 1
        for c in range(parts):
            m = assign == c
            if m.any():
                v = M[m].mean(axis=0)
                norm = float(np.linalg.norm(v))
                if norm > 0.0:
                    cent[c] = v / norm
    return [[members[int(i)] for i in np.where(assign == c)[0]] for c in range(parts)]


def bundle(V: np.ndarray, *, tau: float, k_max: int, min_size: int, rounds: int = 5) -> Bundles:
    """二段：τ の覆いで**自然な群**を作り（近いものだけ）、群の中を容量 k で**同じくらいの大きさ**に割る。

    容量を空間全体に掛けると、満ちた群からあふれた記録が別の群へ流れ込む（丘をまたぐ）。
    割るのは群の中だけにして、それを防ぐ。
    """
    U = _unit_rows(V)
    n = U.shape[0]
    if n == 0:
        return Bundles([], [], [], 0, 0, 0)
    seeds = _seeds(U, tau)
    cover = len(seeds)
    k = max(1, min(int(k_max), -(-n // cover)))
    # 自然な群：最も近い種へ。中心を更新して数回。
    cent = U[seeds].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(max(1, int(rounds))):
        assign = np.argmax(U @ cent.T, axis=1)
        for c in range(cover):
            m = assign == c
            if m.any():
                v = U[m].mean(axis=0)
                norm = float(np.linalg.norm(v))
                if norm > 0.0:
                    cent[c] = v / norm
    sim_to_center = (U * cent[assign]).sum(axis=1)
    isolated = [int(i) for i in np.where(sim_to_center < tau)[0]]
    bundles: list[list[int]] = []
    center_sim: list[float] = []
    dropped = 0
    for c in range(cover):
        members = [int(i) for i in np.where((assign == c) & (sim_to_center >= tau))[0]]
        if not members:
            continue
        for part in _split_with_capacity(U, members, k, rounds):
            if len(part) < int(min_size):
                dropped += len(part)
                continue
            bundles.append(sorted(part))
            center_sim.append(float(sim_to_center[part].mean()))
    return Bundles(bundles, center_sim, isolated, dropped, cover, k, seeds)
