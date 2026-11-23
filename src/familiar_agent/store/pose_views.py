"""定点ごとの前の見えの保存層（`pose_views`・知-i 段 2）。

見えの普通（`pose_norms`）は移動平均で、置き物が変わってもすぐ吸収される。前の見えは定点ごとに 1 枚で、
見比べて「物が変わった」と判定したときだけ置き換える。見比べたが変わっていなかったときは時刻だけ残す。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..db import under_db_lock
from .pose_norms import NORM_DIM


@dataclass(frozen=True)
class EarlierView:
    """前の見え。`checked_at` は最後に見比べた時刻（エポック秒・まだなら None）。"""

    embedding: list[float]
    image: bytes
    checked_at: float | None


def _vector(raw) -> list[float]:
    # pgvector は文字列 "[a,b,…]" で返ることがある。
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return [float(x) for x in raw]


@under_db_lock
class PoseViewStore:
    """`pose_views` の読み書き。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    def load(self, pose: str) -> EarlierView | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding, image, EXTRACT(EPOCH FROM checked_at) AS checked "
                "FROM pose_views WHERE pose_name = %s",
                (pose,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        emb, image, checked = (
            row if isinstance(row, tuple) else (row["embedding"], row["image"], row["checked"])
        )
        return EarlierView(_vector(emb), bytes(image), None if checked is None else float(checked))

    def save(self, pose: str, embedding: list[float], image: bytes) -> None:
        """前の見えを置き換える。置き換えた時を見比べた時にもする（すぐに見比べ直さない）。"""
        if len(embedding) != NORM_DIM:
            raise ValueError(
                f"前の見えは {NORM_DIM} 次元でなければならない（渡されたのは {len(embedding)}）"
            )
        vector = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pose_views (pose_name, embedding, image, seen_at, checked_at)
                VALUES (%s, %s::vector, %s, now(), now())
                ON CONFLICT (pose_name) DO UPDATE
                   SET embedding = EXCLUDED.embedding, image = EXCLUDED.image,
                       seen_at = now(), checked_at = now()
                """,
                (pose, vector, image),
            )
        self._conn.commit()

    def mark_checked(self, pose: str) -> None:
        """見比べたが置き換えなかった（明るさだけ・変わっていない・読めなかった）。時刻だけ残す。"""
        with self._conn.cursor() as cur:
            cur.execute("UPDATE pose_views SET checked_at = now() WHERE pose_name = %s", (pose,))
        self._conn.commit()
