"""課題7 v0.3 実機計測スクリプト
計測1: c_lo/c_hi (平均中心化後コサイン + 意味ラベル関連ペア + ZCA whitening)
計測2: CSV保存
DB接続: familiar_agent.db.get_db() (memory.py と同じ)
"""
from __future__ import annotations
import os, shutil, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

os.environ.setdefault("DATABASE_URL", "postgresql://familiar:familiar@localhost:5432/familiar_ai")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from familiar_agent.db import get_db, sql_to_vec
from familiar_agent.person_memory_manager import AGENT_SELF_ID

RNG = np.random.default_rng(42)
N_UNREL_PAIRS = 50_000
OUT = Path.home() / "Downloads" / "measure_v03"
OUT.mkdir(parents=True, exist_ok=True)

PCTS = [50, 75, 90, 95, 99]
V01_C_LO = 0.9311  # v0.1 生コサイン基準値
V01_C_HI = 0.9475
V01_WINDOW = V01_C_HI - V01_C_LO  # 0.0164

db = get_db()


# ── ユーティリティ ─────────────────────────────────────────────

def pct_table(arr: np.ndarray) -> dict[int, float]:
    return {p: float(np.percentile(arr, p)) for p in PCTS}

def cosine_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return num / den

def unrelated_sample(vecs: np.ndarray, n: int = N_UNREL_PAIRS) -> np.ndarray:
    N = len(vecs)
    i = RNG.integers(0, N, n)
    j = RNG.integers(0, N, n)
    mask = i != j
    return cosine_pairs(vecs[i[mask]], vecs[j[mask]])

def center_and_normalize(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centered_normed, mu)"""
    mu = vecs.mean(axis=0)
    centered = vecs - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9
    return (centered / norms).astype(np.float32), mu.astype(np.float32)

def zca_whiten(vecs: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """ZCA whitening. Returns (whitened, W_matrix)."""
    cov = (vecs.T @ vecs) / len(vecs)
    d, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(d, eps))) @ V.T
    whitened = vecs @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-9
    return (whitened / norms).astype(np.float32), W.astype(np.float32)

def rank_sep(rel: np.ndarray, unrel: np.ndarray) -> float:
    """P(関連>無関係): 全ペアのうち関連コサイン > 無関係コサインの割合."""
    n = min(10_000, len(rel), len(unrel))
    ri = RNG.choice(len(rel),   n, replace=True)
    ui = RNG.choice(len(unrel), n, replace=True)
    return float(np.mean(rel[ri] > unrel[ui]))


# ── DB からベクトルと timestamp を取得 ────────────────────────

def load_vecs_with_ts(pid: str) -> tuple[np.ndarray, list[datetime], list[str]]:
    """situated vectors + observation timestamps + obs_ids"""
    with db.cursor() as cur:
        cur.execute("""
            SELECT s.obs_id, s.vector, o.timestamp
            FROM situated_embeddings s
            JOIN observations o ON s.obs_id = o.id
            WHERE s.person_id = %s
            ORDER BY o.timestamp
        """, (pid,))
        rows = cur.fetchall()
    vecs = np.array([sql_to_vec(r["vector"]) for r in rows], dtype=np.float32)
    ts   = [r["timestamp"] for r in rows]
    oids = [r["obs_id"] for r in rows]
    return vecs, ts, oids


# ── 意味ラベル関連ペア作成 ────────────────────────────────────

def semantic_related_pairs(
    vecs: np.ndarray,
    ts: list[datetime],
    same_day_cap: int = 3000,
    prox_gap_min: int = 10,
    prox_cap: int = 2000,
) -> tuple[np.ndarray, dict]:
    """同一日付ペア + 時間近接ペア（10分以内）でコサインを計算。
    vecs は中心化済みを渡す。"""
    # timestamp を UTC aware に統一
    ts_aware = []
    for t in ts:
        if t is None:
            ts_aware.append(None)
        elif t.tzinfo is None:
            ts_aware.append(t.replace(tzinfo=timezone.utc))
        else:
            ts_aware.append(t)

    # ── 同一日ペア ──────────────────────────────────────────
    from collections import defaultdict
    day_idx: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(ts_aware):
        if t is not None:
            day_idx[t.strftime("%Y-%m-%d")].append(i)

    same_day_pairs: list[tuple[int, int]] = []
    for idxs in day_idx.values():
        if len(idxs) < 2:
            continue
        arr = np.array(idxs)
        ai = RNG.integers(0, len(arr), min(same_day_cap, len(arr) * 4))
        bi = RNG.integers(0, len(arr), min(same_day_cap, len(arr) * 4))
        mask = ai != bi
        for a, b in zip(arr[ai[mask]], arr[bi[mask]]):
            same_day_pairs.append((int(a), int(b)))
            if len(same_day_pairs) >= same_day_cap:
                break
        if len(same_day_pairs) >= same_day_cap:
            break

    # ── 時間近接ペア (10分以内) ────────────────────────────
    prox_pairs: list[tuple[int, int]] = []
    gap = timedelta(minutes=prox_gap_min)
    valid = [(i, t) for i, t in enumerate(ts_aware) if t is not None]
    for idx_in_valid, (i, ti) in enumerate(valid):
        for j, tj in valid[idx_in_valid + 1:]:
            if tj - ti > gap:
                break
            prox_pairs.append((i, j))
            if len(prox_pairs) >= prox_cap:
                break
        if len(prox_pairs) >= prox_cap:
            break

    # ── コサイン計算 ───────────────────────────────────────
    def pairs_to_cosine(pairs):
        if not pairs:
            return np.array([], dtype=np.float32)
        ai = np.array([p[0] for p in pairs])
        bi = np.array([p[1] for p in pairs])
        return cosine_pairs(vecs[ai], vecs[bi])

    cos_day  = pairs_to_cosine(same_day_pairs)
    cos_prox = pairs_to_cosine(prox_pairs)

    combined = np.concatenate([cos_day, cos_prox]) if len(cos_day) + len(cos_prox) > 0 else np.array([], dtype=np.float32)
    # 重複除去のため unique (近似)
    if len(combined) > 0:
        combined = np.unique(combined)

    meta = {
        "same_day_pairs":  len(same_day_pairs),
        "prox_pairs":      len(prox_pairs),
        "total_rel_cosines": len(combined),
        "days_available":  len(day_idx),
    }
    return combined, meta


# ── 各視点の計測 ──────────────────────────────────────────────

def measure_pid(label: str, pid: str) -> dict:
    print(f"\n{'='*60}")
    print(f"視点: {label}  ({pid[:8]}...)")
    print(f"{'='*60}")

    raw_vecs, ts, oids = load_vecs_with_ts(pid)
    N = len(raw_vecs)
    print(f"  ベクトル件数: {N}  dim={raw_vecs.shape[1]}")

    # ── 平均中心化 ────────────────────────────────────────
    c_vecs, mu = center_and_normalize(raw_vecs)
    np.save(OUT / f"mu_{label}.npy", mu)
    print(f"  mu 保存: {OUT}/mu_{label}.npy  |mu|={np.linalg.norm(mu):.4f}")

    # ── 無関係分布 ────────────────────────────────────────
    unrel = unrelated_sample(c_vecs)
    u_pct = pct_table(unrel)
    print(f"  無関係(中心化後) p50={u_pct[50]:.4f} p75={u_pct[75]:.4f} "
          f"p90={u_pct[90]:.4f} p95={u_pct[95]:.4f} p99={u_pct[99]:.4f}")

    # ── 意味ラベル関連ペア ────────────────────────────────
    rel_sem, meta_sem = semantic_related_pairs(c_vecs, ts)
    print(f"  意味ラベルペア: 同一日={meta_sem['same_day_pairs']}  "
          f"近接10分={meta_sem['prox_pairs']}  ユニーク={meta_sem['total_rel_cosines']}")

    # ── 最近傍補助（参考） ────────────────────────────────
    # 中心化後ベクトルで Python 計算（補助として）
    n_query = min(300, N)
    q_idx = RNG.choice(N, n_query, replace=False)
    nn_cos = []
    for qi in q_idx:
        dists = c_vecs @ c_vecs[qi]
        dists[qi] = -1  # 自己除外
        top5 = np.argsort(dists)[-5:]
        nn_cos.extend(dists[top5].tolist())
    rel_nn = np.array(nn_cos, dtype=np.float32)
    nn_pct = pct_table(rel_nn)

    # 主: 意味ラベル（件数が十分なら）、補助: 最近傍
    has_sem = len(rel_sem) >= 100
    rel_main = rel_sem if has_sem else rel_nn
    rel_label = "semantic" if has_sem else "nearest_neighbor(補助)"

    r_pct = pct_table(rel_main)
    print(f"  関連({rel_label}) p50={r_pct[50]:.4f} p75={r_pct[75]:.4f} "
          f"p90={r_pct[90]:.4f} p95={r_pct[95]:.4f} p99={r_pct[99]:.4f}")
    if has_sem:
        print(f"  (参考:最近傍) p50={nn_pct[50]:.4f} p75={nn_pct[75]:.4f} "
              f"p90={nn_pct[90]:.4f} p95={nn_pct[95]:.4f} p99={nn_pct[99]:.4f}")

    c_lo = float(np.percentile(unrel,     95))
    c_hi = float(np.percentile(rel_main,  25))
    window = c_hi - c_lo
    sep    = rank_sep(rel_main, unrel)
    overlap = float(np.mean(rel_main < c_lo)) if len(rel_main) else float("nan")
    print(f"  提案 c_lo={c_lo:.4f}  c_hi={c_hi:.4f}  窓幅={window:.4f}  "
          f"P(関連>無関係)={sep:.3f}  重なり={overlap:.1%}")
    print(f"  窓幅変化: v0.1={V01_WINDOW:.4f} → v0.3={window:.4f}  "
          f"(Δ={window-V01_WINDOW:+.4f}, {(window/V01_WINDOW-1)*100:+.0f}%)")

    # ── ZCA whitening（任意・改善案） ────────────────────
    w_vecs, W = zca_whiten(c_vecs)
    np.save(OUT / f"W_zca_{label}.npy", W)
    unrel_w = unrelated_sample(w_vecs)
    rel_w_sem, _ = semantic_related_pairs(w_vecs, ts)
    rel_w = rel_w_sem if len(rel_w_sem) >= 100 else np.array([w_vecs[q] @ w_vecs for q in q_idx[:50]]).flatten()
    uw_pct = pct_table(unrel_w)
    rw_pct = pct_table(rel_w) if len(rel_w) else {}
    c_lo_w = float(np.percentile(unrel_w, 95))
    c_hi_w = float(np.percentile(rel_w,   25)) if len(rel_w) else float("nan")
    window_w = c_hi_w - c_lo_w if not np.isnan(c_hi_w) else float("nan")
    print(f"  [ZCA] c_lo={c_lo_w:.4f}  c_hi={c_hi_w:.4f}  窓幅={window_w:.4f}")

    # ── CSV保存 ───────────────────────────────────────────
    np.savetxt(OUT / f"unrel_{label}.csv",  unrel,    delimiter=",", header="cosine", comments="")
    np.savetxt(OUT / f"rel_sem_{label}.csv", rel_sem if len(rel_sem) else np.array([0.0]),
               delimiter=",", header="cosine_semantic", comments="")
    np.savetxt(OUT / f"rel_nn_{label}.csv",  rel_nn,  delimiter=",", header="cosine_nn", comments="")
    np.savetxt(OUT / f"unrel_zca_{label}.csv", unrel_w, delimiter=",", header="cosine_zca", comments="")

    return {
        "label": label, "pid": pid, "N": N,
        "u_pct": u_pct, "r_pct": r_pct, "r_label": rel_label,
        "nn_pct": nn_pct,
        "c_lo": c_lo, "c_hi": c_hi, "window": window,
        "sep": sep, "overlap": overlap,
        "sem_meta": meta_sem,
        "c_lo_w": c_lo_w, "c_hi_w": c_hi_w, "window_w": window_w,
        "uw_pct": uw_pct, "rw_pct": rw_pct,
        "mu_path": str(OUT / f"mu_{label}.npy"),
        "W_path":  str(OUT / f"W_zca_{label}.npy"),
    }


# ── 人物IDの解決 ──────────────────────────────────────────────
with db.cursor() as cur:
    cur.execute("""
        SELECT person_id, COUNT(*) as cnt
        FROM situated_embeddings GROUP BY person_id ORDER BY cnt DESC LIMIT 1
    """)
    most_pop_pid = str(cur.fetchone()["person_id"])

    cur.execute("""
        SELECT id, name, display_name FROM persons
        WHERE name NOT IN ('default','__self__') ORDER BY created_at LIMIT 1
    """)
    real = cur.fetchone()
    real_pid  = str(real["id"])
    real_name = real["name"]

print(f"most_populated  : {most_pop_pid[:8]}...")
print(f"実在person      : {real_pid[:8]}... ({real_name})")
print(f"AGENT_SELF      : {AGENT_SELF_ID[:8]}...")

targets = [
    ("most_populated", most_pop_pid),
    ("real_person",    real_pid),
    ("AGENT_SELF",     AGENT_SELF_ID),
]

results = []
for label, pid in targets:
    r = measure_pid(label, pid)
    results.append(r)

# ── スクリプト自身を保存（成果物） ────────────────────────────
shutil.copy(__file__, OUT / "measure_c_lo_hi_v03.py")

# ── サマリ ────────────────────────────────────────────────────
print("\n" + "="*60)
print("集計サマリ")
print("="*60)
print(f"{'視点':<18} {'c_lo':>7} {'c_hi':>7} {'窓幅':>8} {'P(rel>unrel)':>13} {'重なり':>7} {'窓幅(ZCA)':>10}")
for r in results:
    ww = f"{r['window_w']:.4f}" if not np.isnan(r.get('window_w', float('nan'))) else "  N/A "
    print(f"{r['label']:<18} {r['c_lo']:>7.4f} {r['c_hi']:>7.4f} {r['window']:>8.4f} "
          f"{r['sep']:>13.3f} {r['overlap']:>6.1%} {ww:>10}")

print(f"\nv0.1 生コサイン窓幅: {V01_WINDOW:.4f}")
avg_w = np.mean([r["window"] for r in results])
print(f"v0.3 中心化後平均窓幅: {avg_w:.4f}  (Δ={avg_w-V01_WINDOW:+.4f})")
print(f"\n成果物: {OUT}")
