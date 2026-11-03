"""Ruri の数値の表 `obs_ruri_embeddings` を足す（記-y・2026-11-03・本人の決定）。

想起の 2 つめの頭 Ruri（`cl-nagoya/ruri-v3-310m`・768 次元）。数値は**記録の本文**から作るので、
記録 1 件に 1 つ。面（`situated_memories`）ごとには持たない。記録の番号を鍵にし、記録が消えれば
一緒に消える（`obs_embeddings` と同じ形）。平均は引かない。

行が無い記録は「Ruri の数値がまだ無い」で、想起は bge-m3 だけで採点する。数値にできなかったときに
0 を書かないので、列は NOT NULL にする（空は行が無いことで表す）。

今ある記録はこの表を作っただけでは埋まらない。`scripts/backfill_ruri.py` で埋める。

冪等（`IF NOT EXISTS`）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS obs_ruri_embeddings (
                obs_id  TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
                vector  vector(768) NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_ruri_hnsw ON obs_ruri_embeddings "
            "USING hnsw (vector vector_cosine_ops)"
        )
