#!/usr/bin/env python3
"""bge-m3 移行後の mu（平均中心化ベクトル）再計算スクリプト。

situated_embeddings（bge-m3 / 1024次元）から person_id 別の平均ベクトルを
計算して ~/Downloads/measure_mu_bge_m3/ に保存する。

計測指示書 v0.5 の c_lo/c_hi 計測（ステップ8）で使用する。

Usage:
    uv run python scripts/compute_mu.py
    uv run python scripts/compute_mu.py --db-url postgresql://...
    uv run python scripts/compute_mu.py --out-dir /path/to/dir
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://familiar:familiar@localhost:5432/familiar_ai",
)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from familiar_agent.db import sql_to_vec
from familiar_agent.person_memory_manager import AGENT_SELF_ID


def _conn(db_url: str):
    c = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


def load_situated_vecs(conn, person_id: str) -> np.ndarray:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.vector
            FROM situated_embeddings s
            JOIN observations o ON s.obs_id = o.id
            WHERE s.person_id = %s
              AND o.superseded_by IS NULL
            ORDER BY o.timestamp
        """, (person_id,))
        rows = cur.fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.array([sql_to_vec(r["vector"]) for r in rows], dtype=np.float32)


def load_persons(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.id, p.name,
                   COUNT(s.id) AS vec_count
            FROM persons p
            LEFT JOIN situated_embeddings s ON s.person_id = p.id
            GROUP BY p.id, p.name
            ORDER BY vec_count DESC
        """)
        return cur.fetchall()


def safe_label(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Downloads" / "measure_mu_bge_m3"),
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"出力先: {out}")

    conn = _conn(args.db_url)
    persons = load_persons(conn)

    print(f"\n登録 person 数: {len(persons)}")
    print(f"{'name':<20} {'id':>38} {'vecs':>6}")
    print("-" * 68)
    for p in persons:
        marker = " ← AGENT_SELF" if p["id"] == AGENT_SELF_ID else ""
        print(f"{p['name']:<20} {p['id']:>38} {p['vec_count']:>6}{marker}")

    print()
    summary = []

    for p in persons:
        pid   = p["id"]
        name  = p["name"]
        label = safe_label(name) if name not in ("default", "__self__") else (
            "AGENT_SELF" if pid == AGENT_SELF_ID else safe_label(name)
        )

        vecs = load_situated_vecs(conn, pid)
        n = len(vecs)

        if n < 2:
            print(f"  [{label}] ベクトル {n}件 — スキップ（2件未満）")
            continue

        mu = vecs.mean(axis=0).astype(np.float32)
        mu_norm = float(np.linalg.norm(mu))

        fname = out / f"mu_{label}.npy"
        np.save(fname, mu)

        print(f"  [{label}]  n={n}  dim={vecs.shape[1]}  |mu|={mu_norm:.4f}  → {fname.name}")
        summary.append({
            "label": label,
            "person_id": pid,
            "n_vecs": n,
            "dim": int(vecs.shape[1]),
            "mu_norm": mu_norm,
            "path": str(fname),
        })

    conn.close()

    print(f"\n完了: {len(summary)} 件の mu を保存しました。")
    print(f"保存先: {out}")

    # 次ステップ案内
    print("\n次のステップ:")
    print("  計測指示書 v0.5 の計測1（c_lo/c_hi）を bge-m3 で実施してください。")
    print("  mu ファイルは上記ディレクトリを参照してください。")


if __name__ == "__main__":
    main()
