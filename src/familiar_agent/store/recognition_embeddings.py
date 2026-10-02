"""顔と声の特徴の器（知-ae 段 1・2026-10-02・`設計方針_声で話者を見分ける` v0.1）。表 `recognition_embeddings`（073）。

顔と声を 1 つの表に、種類（`face`・`voice`）と出どころ（`registered`・`today`）で分けて持つ（本人の決定ア）。

- **照らすのは人ごとの重心**（正規化した特徴の平均）。1 つずつ比べて最も近いものを取ると、たまたま似た 1 つに
  引かれて取り違えが増える。
- **上限を超えたら古いものから捨てる**（`cap`。登録の声 30・今日の声 10 は `RecognitionConfig` が持つ仮の値）。
- **今日の声は `day` の日だけ引く**。前の日の分は `drop_old_today` で捨てる。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import psycopg2.extras

KINDS = ("face", "voice")
ORIGINS = ("registered", "today")


class RecognitionEmbeddingStore:
    def __init__(self, conn) -> None:
        self._conn = conn

    def add(
        self,
        person_id: str,
        kind: str,
        origin: str,
        vec: np.ndarray,
        *,
        cap: int,
        day: "date | None" = None,
        now: "datetime | None" = None,
    ) -> None:
        """特徴を 1 つ足し、その人・種類・出どころで `cap` を超えた古いものを捨てる。"""
        if kind not in KINDS or origin not in ORIGINS:
            raise ValueError(f"種類か出どころが違う：{kind}・{origin}")
        now = now or datetime.now(timezone.utc)
        values = [float(x) for x in np.asarray(vec, dtype=np.float32).ravel()]
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO recognition_embeddings (person_id, kind, origin, day, vec, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (person_id, kind, origin, day, values, now),
            )
            cur.execute(
                "DELETE FROM recognition_embeddings WHERE id IN ("
                " SELECT id FROM recognition_embeddings"
                " WHERE person_id = %s AND kind = %s AND origin = %s"
                " ORDER BY created_at DESC, id DESC OFFSET %s)",
                (person_id, kind, origin, max(0, int(cap))),
            )
        self._commit()

    def centroids(
        self, kind: str, origin: str, *, day: "date | None" = None
    ) -> "dict[str, np.ndarray]":
        """人ごとの重心（正規化した特徴の平均）。今日の声は `day` の日の分だけ。"""
        sql = "SELECT person_id, vec FROM recognition_embeddings WHERE kind = %s AND origin = %s"
        args: list = [kind, origin]
        if day is not None:
            sql += " AND day = %s"
            args.append(day)
        groups: dict[str, list[np.ndarray]] = {}
        with self._cursor() as cur:
            cur.execute(sql, args)
            for r in cur.fetchall():
                v = np.asarray(r["vec"], dtype=np.float32)
                n = float(np.linalg.norm(v))
                if n > 0.0:
                    groups.setdefault(str(r["person_id"]), []).append(v / n)
        return {pid: np.mean(vs, axis=0).astype(np.float32) for pid, vs in groups.items()}

    def count(self, person_id: str, kind: str, origin: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM recognition_embeddings "
                "WHERE person_id = %s AND kind = %s AND origin = %s",
                (person_id, kind, origin),
            )
            return int(cur.fetchone()["n"])

    def drop_old_today(self, today: date) -> int:
        """前の日までの今日の声を捨てる。捨てた数を返す。"""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM recognition_embeddings WHERE origin = 'today' "
                "AND (day IS NULL OR day < %s)",
                (today,),
            )
            n = cur.rowcount
        self._commit()
        return int(n or 0)

    def _cursor(self):
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _commit(self) -> None:
        if not getattr(self._conn, "autocommit", False):
            self._conn.commit()
