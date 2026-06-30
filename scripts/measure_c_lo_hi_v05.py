#!/usr/bin/env python3
"""計測指示書 v0.5 計測1: c_lo/c_hi（中心化後コサイン、擬似セッション関連）.

読み取り専用 — INSERT/UPDATE/DELETE 禁止。
"""
from __future__ import annotations

import os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

os.environ.setdefault("DATABASE_URL", "postgresql://familiar:familiar@localhost:5432/familiar_ai")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from familiar_agent.db import sql_to_vec
from familiar_agent.person_memory_manager import AGENT_SELF_ID

RNG = np.random.default_rng(42)
N_UNREL = 50_000
MU_DIR  = Path.home() / "Downloads" / "measure_mu_bge_m3"
PCTS    = [25, 50, 75, 90, 95, 99]
SESSION_GAPS = [timedelta(minutes=10), timedelta(minutes=15)]
MAX_PAIRS_PER_SESSION = 500   # セッションが大きい場合のペア数上限


# ── DB ───────────────────────────────────────────────────────────────────────

def get_conn():
    c = psycopg2.connect(
        os.environ["DATABASE_URL"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    c.autocommit = True
    return c


def load_obs_for_person(conn, pid: str) -> list[dict]:
    """situated vector + timestamp + content + kind を時系列順に返す."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.obs_id, s.vector, o.timestamp, o.content, o.kind
            FROM situated_embeddings s
            JOIN observations o ON s.obs_id = o.id
            WHERE s.person_id = %s
              AND o.superseded_by IS NULL
            ORDER BY o.timestamp ASC
        """, (pid,))
        return cur.fetchall()


def load_persons(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.id, p.name, COUNT(s.id) AS n
            FROM persons p
            JOIN situated_embeddings s ON s.person_id = p.id
            GROUP BY p.id, p.name
            ORDER BY n DESC
        """)
        return cur.fetchall()


# ── ベクトル操作 ─────────────────────────────────────────────────────────────

def center_normalize(vecs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    c = vecs - mu
    norms = np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
    return (c / norms).astype(np.float32)


def cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行ペア (a[i], b[i]) のコサイン類似度（正規化済みベクトル前提）."""
    return np.clip((a * b).sum(axis=1), -1.0, 1.0)


# ── 擬似セッション ───────────────────────────────────────────────────────────

def to_utc(t):
    if t is None:
        return None
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    return t


def build_sessions(rows: list[dict], gap: timedelta) -> list[list[int]]:
    """timestamp 間隔が gap 超で別セッション。インデックスのリストを返す。"""
    if not rows:
        return []
    sessions: list[list[int]] = []
    cur_session = [0]
    prev_ts = to_utc(rows[0]["timestamp"])
    for i in range(1, len(rows)):
        ts = to_utc(rows[i]["timestamp"])
        if ts is None or prev_ts is None or (ts - prev_ts) > gap:
            sessions.append(cur_session)
            cur_session = [i]
        else:
            cur_session.append(i)
        prev_ts = ts
    sessions.append(cur_session)
    return sessions


def session_stats(sessions: list[list[int]], rows: list[dict]) -> dict:
    lens = [len(s) for s in sessions if len(s) >= 2]
    if not lens:
        return {"n_sessions": len(sessions), "n_usable": 0,
                "avg_len": 0.0, "median_len": 0.0, "avg_duration_min": 0.0}
    durations = []
    for s in sessions:
        if len(s) < 2:
            continue
        t0 = to_utc(rows[s[0]]["timestamp"])
        t1 = to_utc(rows[s[-1]]["timestamp"])
        if t0 and t1:
            durations.append((t1 - t0).total_seconds() / 60)
    return {
        "n_sessions": len(sessions),
        "n_usable":   len(lens),
        "avg_len":    float(np.mean(lens)),
        "median_len": float(np.median(lens)),
        "avg_duration_min": float(np.mean(durations)) if durations else 0.0,
    }


def related_pairs(
    sessions: list[list[int]],
    vecs_c: np.ndarray,
    rows: list[dict],
) -> np.ndarray:
    """同一セッション内の非重複ペアのコサイン類似度。"""
    cosines: list[float] = []
    for s in sessions:
        if len(s) < 2:
            continue
        # 完全重複（同 content）を除外
        content_set: dict[str, int] = {}  # content -> first idx in session
        deduped = []
        for i in s:
            c = rows[i]["content"]
            if c not in content_set:
                content_set[c] = i
                deduped.append(i)
        if len(deduped) < 2:
            continue
        # ペア列挙（大きいセッションはランダムサンプリング）
        pairs: list[tuple[int, int]] = []
        for a in range(len(deduped)):
            for b in range(a + 1, len(deduped)):
                pairs.append((deduped[a], deduped[b]))
        if len(pairs) > MAX_PAIRS_PER_SESSION:
            idx = RNG.choice(len(pairs), MAX_PAIRS_PER_SESSION, replace=False)
            pairs = [pairs[i] for i in idx]
        ai = np.array([p[0] for p in pairs])
        bi = np.array([p[1] for p in pairs])
        cosines.extend(cosine_batch(vecs_c[ai], vecs_c[bi]).tolist())
    return np.array(cosines, dtype=np.float32)


def unrelated_sample(vecs_c: np.ndarray, n: int = N_UNREL) -> np.ndarray:
    N = len(vecs_c)
    i = RNG.integers(0, N, n)
    j = RNG.integers(0, N, n)
    mask = i != j
    i, j = i[mask], j[mask]
    return cosine_batch(vecs_c[i], vecs_c[j])


# ── 最近傍 top-5 分布 ────────────────────────────────────────────────────────

def nn_top5_distribution(vecs_c: np.ndarray, sample: int = 500) -> np.ndarray:
    """各ベクトルの上位5近傍コサイン（self除外）を sample 件からサンプリング。"""
    N = len(vecs_c)
    sample = min(sample, N)
    idx = RNG.choice(N, sample, replace=False)
    cosines: list[float] = []
    for i in idx:
        sims = vecs_c @ vecs_c[i]   # (N,) — 全コサイン
        sims[i] = -2.0              # self 除外
        top5 = np.partition(sims, -5)[-5:]
        cosines.extend(top5.tolist())
    return np.array(cosines, dtype=np.float32)


# ── パーセンタイル表 ─────────────────────────────────────────────────────────

def pct_table(arr: np.ndarray) -> dict[int, float]:
    if len(arr) == 0:
        return {p: float("nan") for p in PCTS}
    return {p: float(np.percentile(arr, p)) for p in PCTS}


def rank_sep(rel: np.ndarray, unrel: np.ndarray, n: int = 10_000) -> float:
    if len(rel) == 0 or len(unrel) == 0:
        return float("nan")
    n = min(n, len(rel), len(unrel))
    ri = RNG.choice(len(rel),   n, replace=True)
    ui = RNG.choice(len(unrel), n, replace=True)
    return float(np.mean(rel[ri] > unrel[ui]))


def overlap_rate(rel: np.ndarray, unrel: np.ndarray, c_lo: float) -> float:
    """関連のうち c_lo 未満の割合（veto 率）."""
    if len(rel) == 0:
        return float("nan")
    return float(np.mean(rel < c_lo))


# ── メイン計測 ───────────────────────────────────────────────────────────────

def measure_person(label: str, pid: str, conn) -> None:
    print(f"\n{'='*65}")
    print(f"視点: {label}  ({pid[:8]}...)")
    print(f"{'='*65}")

    # mu ロード
    mu_path = MU_DIR / f"mu_{label}.npy"
    if not mu_path.exists():
        # ファイル名が person name と一致しない可能性がある → fallback
        candidates = list(MU_DIR.glob("mu_*.npy"))
        print(f"  [警告] {mu_path.name} が見つかりません。候補: {[p.name for p in candidates]}")
        return
    mu = np.load(mu_path).astype(np.float32)
    print(f"  mu: {mu_path.name}  shape={mu.shape}  |mu|={np.linalg.norm(mu):.4f}")

    # データロード
    rows = load_obs_for_person(conn, pid)
    N = len(rows)
    print(f"  observations: {N}  dim={mu.shape[0]}")
    if N < 50:
        print("  データ不足 — スキップ")
        return

    raw_vecs = np.array([sql_to_vec(r["vector"]) for r in rows], dtype=np.float32)
    vecs_c   = center_normalize(raw_vecs, mu)

    # 無関係分布
    unrel = unrelated_sample(vecs_c)
    u_pct = pct_table(unrel)
    print(f"\n  【無関係分布】n={len(unrel):,}")
    print(f"    p50={u_pct[50]:.4f}  p75={u_pct[75]:.4f}  p90={u_pct[90]:.4f}"
          f"  p95={u_pct[95]:.4f}  p99={u_pct[99]:.4f}")

    # 最近傍 top-5 分布
    nn5 = nn_top5_distribution(vecs_c)
    nn5_pct = pct_table(nn5)
    print("\n  【最近傍 top-5 分布】(sample=500 vecs)")
    print(f"    p25={nn5_pct[25]:.4f}  p50={nn5_pct[50]:.4f}  p75={nn5_pct[75]:.4f}"
          f"  p90={nn5_pct[90]:.4f}  p95={nn5_pct[95]:.4f}")

    # 擬似セッション × ギャップ閾値
    for gap in SESSION_GAPS:
        gmin = int(gap.total_seconds() // 60)
        print(f"\n  ── セッション定義: gap > {gmin}分 ──────────────────────────────")

        sessions = build_sessions(rows, gap)
        stats = session_stats(sessions, rows)
        print(f"    セッション数={stats['n_sessions']}  "
              f"2件以上={stats['n_usable']}  "
              f"平均発話数={stats['avg_len']:.1f}  "
              f"中央値発話数={stats['median_len']:.1f}  "
              f"平均時間長={stats['avg_duration_min']:.1f}分")

        rel = related_pairs(sessions, vecs_c, rows)
        r_pct = pct_table(rel)
        print(f"\n    【関連分布】n={len(rel):,}")
        if len(rel) == 0:
            print("    関連ペアなし — スキップ")
            continue
        print(f"    p25={r_pct[25]:.4f}  p50={r_pct[50]:.4f}  p75={r_pct[75]:.4f}"
              f"  p90={r_pct[90]:.4f}  p95={r_pct[95]:.4f}  p99={r_pct[99]:.4f}")

        c_lo = u_pct[95]
        c_hi = r_pct[25]
        window = c_hi - c_lo
        p_sep = rank_sep(rel, unrel)
        veto  = overlap_rate(rel, unrel, c_lo)

        print("\n    【提案値】")
        print(f"    c_lo (無関係 p95) = {c_lo:.4f}")
        print(f"    c_hi (関連  p25)  = {c_hi:.4f}")
        print(f"    窓幅              = {window:+.4f}")
        print(f"    P(関連>無関係)    = {p_sep:.3f}")
        print(f"    veto 率(関連<c_lo)= {veto:.1%}")

        # 前回比較 (e5-small)
        prev_clo, prev_chi = 0.354, 0.555
        print("\n    【前回比（e5-small）】")
        print(f"    c_lo: {prev_clo:.4f} → {c_lo:.4f}  (Δ={c_lo-prev_clo:+.4f})")
        print(f"    c_hi: {prev_chi:.4f} → {c_hi:.4f}  (Δ={c_hi-prev_chi:+.4f})")
        print(f"    窓幅: {prev_chi-prev_clo:.4f} → {window:.4f}")
        print(f"    前回 veto 率 64% → 今回 {veto:.1%}")


# ── エントリポイント ─────────────────────────────────────────────────────────

def main() -> None:
    print("計測指示書 v0.5 計測1: c_lo/c_hi（bge-m3 / 中心化後コサイン）")
    print(f"mu ディレクトリ: {MU_DIR}")
    print(f"乱数シード: 42  無関係ペア数: {N_UNREL:,}  セッション閾値: 10分・15分")

    conn = get_conn()
    persons = load_persons(conn)

    print(f"\nDB: {len(persons)} persons × situated_embeddings")
    for p in persons:
        print(f"  {p['name']:<20} {p['id'][:8]}...  n={p['n']}")

    # 計測対象を特定
    most_pop  = persons[0]   # n が最大（全員同数の場合は先頭）
    yuusuke   = next((p for p in persons if "ゆうすけ" in (p["name"] or "")), None)
    agent_self = next((p for p in persons if p["id"] == AGENT_SELF_ID), None)

    if yuusuke is None:
        print("[警告] ゆうすけ が見つかりません")
    if agent_self is None:
        print("[警告] AGENT_SELF が見つかりません")

    targets = [
        ("most_populated", most_pop["name"], most_pop["id"]),
    ]
    if yuusuke:
        targets.append(("いくながゆうすけ", yuusuke["name"], yuusuke["id"]))
    if agent_self:
        targets.append(("AGENT_SELF", "__self__", agent_self["id"]))

    for mu_label, _name, pid in targets:
        measure_person(mu_label, pid, conn)

    conn.close()
    print(f"\n\n{'='*65}")
    print("計測完了")
    print(f"mu ファイル保存先: {MU_DIR}")


if __name__ == "__main__":
    main()
