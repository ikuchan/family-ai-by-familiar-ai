#!/usr/bin/env python3
"""Batch re-embed all observations with bge-m3 (migration 020 follow-up).

Run AFTER migration 020 has been applied. Re-encodes every non-superseded
observation and upserts obs_embeddings (BYTEA) and situated_embeddings
(vector(1024)) for every registered person + AGENT_SELF.

Safe to re-run: all upserts use ON CONFLICT DO UPDATE. Interrupted runs
can be resumed; already-written rows are overwritten idempotently.

Usage:
    uv run python scripts/reembed_all.py
    uv run python scripts/reembed_all.py --db-url postgresql://user:pass@host/db
    uv run python scripts/reembed_all.py --batch-size 32
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from familiar_agent.db import vec_to_sql
from familiar_agent.person_memory_manager import AGENT_SELF_ID, ALPHA
from familiar_agent.tools.memory import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    _EmbeddingModel,
    _encode_vector,
    _normalise,
)

_DEFAULT_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://familiar:familiar@localhost:5432/familiar_ai",
)


def _conn(db_url: str):
    c = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = False
    return c


def _load_observations(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM observations "
            "WHERE superseded_by IS NULL "
            "ORDER BY timestamp ASC"
        )
        return cur.fetchall()


def _load_persons(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM persons")
        ids = [r["id"] for r in cur.fetchall()]
    if AGENT_SELF_ID not in ids:
        ids.append(AGENT_SELF_ID)
    return ids


def _zero_perspective_vecs(person_ids: list[str]) -> dict[str, np.ndarray]:
    # Old perspective vectors are 384-dim (e5-small) and meaningless in bge-m3
    # space. Use zeros; the app rebuilds them naturally as new memories are written.
    return {pid: np.zeros(EMBEDDING_DIM, dtype=np.float32) for pid in person_ids}


def _reset_perspective_vecs_in_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE persons SET perspective_vec = NULL")
    conn.commit()


def _write_batch(
    conn,
    obs_ids: list[str],
    vecs: list[list[float]],
    person_ids: list[str],
    p_vecs: dict[str, np.ndarray],
) -> None:
    with conn.cursor() as cur:
        for obs_id, vec in zip(obs_ids, vecs):
            mem_vec = np.array(vec, dtype=np.float32)
            blob = _encode_vector(vec)

            cur.execute(
                "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s) "
                "ON CONFLICT (obs_id) DO UPDATE SET vector = EXCLUDED.vector",
                (obs_id, blob),
            )

            for pid in person_ids:
                situated = _normalise(mem_vec + ALPHA * p_vecs[pid])
                cur.execute(
                    "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) "
                    "VALUES (%s, %s, %s, %s::vector) "
                    "ON CONFLICT (obs_id, person_id) DO UPDATE SET vector = EXCLUDED.vector",
                    (str(uuid.uuid4()), obs_id, pid, vec_to_sql(situated.tolist())),
                )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=_DEFAULT_DB_URL)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="observations per encoding batch (default: 64)")
    args = parser.parse_args()

    print(f"DB : {args.db_url}")
    conn = _conn(args.db_url)

    obs = _load_observations(conn)
    total = len(obs)
    print(f"Observations (non-superseded) : {total}")

    person_ids = _load_persons(conn)
    print(f"Persons (incl. AGENT_SELF)    : {len(person_ids)}")

    p_vecs = _zero_perspective_vecs(person_ids)
    print("Perspective vectors          : zeroed (old 384-dim vectors are stale)")

    print(f"Embedding model               : {EMBEDDING_MODEL}")
    print("Loading model (first run downloads ~1.5 GB) ...")
    embedder = _EmbeddingModel()
    # Trigger synchronous load by encoding a dummy text
    embedder.encode_document(["warmup"])
    if embedder._failed:
        print("ERROR: failed to load embedding model. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("Model ready.")

    batch_size = args.batch_size
    done = 0

    for start in range(0, total, batch_size):
        batch = obs[start : start + batch_size]
        batch_obs_ids = [r["id"] for r in batch]
        batch_contents = [r["content"] for r in batch]

        batch_vecs = embedder.encode_document(batch_contents)
        _write_batch(conn, batch_obs_ids, batch_vecs, person_ids, p_vecs)

        done += len(batch)
        bar = "#" * (done * 40 // total)
        print(f"  [{bar:<40}] {done}/{total}", end="\r", flush=True)

    print(f"\nRe-embedding complete: {done} observations, "
          f"{done * len(person_ids)} situated vectors written.")

    print("Resetting perspective vectors in persons table ...")
    _reset_perspective_vecs_in_db(conn)
    print("Done.")
    conn.close()


if __name__ == "__main__":
    main()
