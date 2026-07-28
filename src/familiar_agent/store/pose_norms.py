"""定点ごとの「見えの普通」の保存層（`pose_norms`）。

`知覚在席` §3-4 の見え層。定点ごとに DINOv2 埋め込み（384次元）の EMA を持つ。行数は定点の
数しかない（実機で3）ので、定点名で1件ずつ引いて上書きするだけでよい。

部屋の映像から作る値なので、`agent_state` の雑多なキーバリューには混ぜず専用のテーブルに置く。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# `facebook/dinov2-small`（ViT-S/14）の出力次元。テーブルの列と揃える。
NORM_DIM = 384


class PoseNormStore:
    """`pose_norms` の読み書き。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    def load(self, pose: str) -> tuple[list[float] | None, int]:
        """その定点の「普通」と、何回見たか。まだ無ければ `(None, 0)`。"""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding, observations FROM pose_norms WHERE pose_name = %s",
                (pose,),
            )
            row = cur.fetchone()
        if row is None:
            return (None, 0)
        raw, observations = (row[0], row[1]) if isinstance(row, tuple) else (
            row["embedding"], row["observations"])
        if raw is None:
            return (None, int(observations))
        # pgvector は文字列 "[a,b,…]" で返ることがある。
        if isinstance(raw, str):
            return ([float(x) for x in raw.strip("[]").split(",")], int(observations))
        return ([float(x) for x in raw], int(observations))

    def save(self, pose: str, embedding: list[float], *, observations: int) -> None:
        """その定点の「普通」を置き換える。

        次元が違う値を混ぜると距離が意味を失うので、入る前に弾く（モデルを替えたときに
        黙って壊れないようにする）。
        """
        if len(embedding) != NORM_DIM:
            raise ValueError(
                f"見えの普通は {NORM_DIM} 次元でなければならない（渡されたのは {len(embedding)}）"
            )
        vector = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pose_norms (pose_name, embedding, observations, updated_at)
                VALUES (%s, %s::vector, %s, now())
                ON CONFLICT (pose_name) DO UPDATE
                   SET embedding = EXCLUDED.embedding,
                       observations = EXCLUDED.observations,
                       updated_at = now()
                """,
                (pose, vector, int(observations)),
            )
        self._conn.commit()
