"""BUG-1: Purge duplicate utterance observations.

Same (content, kind) written multiple times within 60 seconds — keeps the
earliest per cluster, marks the rest superseded_by = earliest id, then
removes their embeddings so recall is unaffected.

Run after: 2026-06-14-018_pending_speech.py
"""
from __future__ import annotations
from datetime import timezone


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # Collect groups with more than one unsuperseded observation
        # sharing the same (content, kind).
        cur.execute("""
            SELECT content, kind,
                   array_agg(id        ORDER BY timestamp ASC, id ASC) AS ids,
                   array_agg(timestamp ORDER BY timestamp ASC, id ASC) AS timestamps
            FROM observations
            WHERE superseded_by IS NULL
            GROUP BY content, kind
            HAVING COUNT(*) > 1
        """)
        groups = cur.fetchall()

    to_supersede: dict[str, str] = {}  # dup_id -> keep_id

    for row in groups:
        ids = row["ids"]
        timestamps = row["timestamps"]

        # Normalise timestamps to UTC-aware for comparison
        ts_aware = []
        for t in timestamps:
            if t is None:
                ts_aware.append(None)
            elif t.tzinfo is None:
                ts_aware.append(t.replace(tzinfo=timezone.utc))
            else:
                ts_aware.append(t)

        # Sliding-window cluster: observations within 60 s of the cluster
        # start are duplicates of it.
        cluster_start = 0
        for i in range(1, len(ids)):
            if ts_aware[i] is None or ts_aware[cluster_start] is None:
                cluster_start = i
                continue
            dt = (ts_aware[i] - ts_aware[cluster_start]).total_seconds()
            if dt <= 60:
                to_supersede[ids[i]] = ids[cluster_start]
            else:
                cluster_start = i

    if not to_supersede:
        return

    with conn.cursor() as cur:
        # Mark duplicates as superseded
        for dup_id, keep_id in to_supersede.items():
            cur.execute(
                "UPDATE observations SET superseded_by = %s WHERE id = %s",
                (keep_id, dup_id),
            )

        # Remove embeddings for superseded rows so they don't occupy recall
        cur.execute("""
            DELETE FROM situated_embeddings
            WHERE obs_id IN (
                SELECT id FROM observations WHERE superseded_by IS NOT NULL
            )
        """)
        cur.execute("""
            DELETE FROM obs_embeddings
            WHERE obs_id IN (
                SELECT id FROM observations WHERE superseded_by IS NOT NULL
            )
        """)
