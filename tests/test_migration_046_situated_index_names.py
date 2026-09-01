"""046（索引名・制約名を表の改名へ追随させる）が効いていることを確かめる。

PostgreSQL の `ALTER TABLE ... RENAME TO ...` は**索引名も制約名も変えない**。044 で
`situated_embeddings` を `situated_memories` へ改めたあと、名前だけが旧いまま残っていた。

**挙動は変わらない。** 索引名は問い合わせ計画に影響しない。復元されたスキーマを
2026-08-21 のダンプと一字一句そろえる段である。

期待する名前は、復元した本番 DB（8月21日）の実物から取った。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]

# 2026-08-21 のダンプに残っていた名前（`pg_indexes` / `pg_constraint` から採取）
_EXPECTED_INDEXES = {
    "idx_situated_hnsw",
    "idx_situated_person",
    "idx_situated_recency",
    "situated_memories_obs_person_relation_key",
    "situated_memories_pkey",
}
_EXPECTED_CONSTRAINTS = {
    "situated_memories_obs_id_fkey",
    "situated_memories_obs_person_relation_key",
    "situated_memories_person_id_fkey",
    "situated_memories_pkey",
}


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _indexes() -> set[str]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename='situated_memories'"
            )
            return {r["indexname"] for r in cur.fetchall()}
    finally:
        conn.close()


def _constraints() -> set[str]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'situated_memories'::regclass"
            )
            return {r["conname"] for r in cur.fetchall()}
    finally:
        conn.close()


def test_the_index_names_follow_the_table() -> None:
    got = _indexes()
    assert got == _EXPECTED_INDEXES, (
        f"欠け: {_EXPECTED_INDEXES - got}\n余り: {got - _EXPECTED_INDEXES}"
    )


def test_the_constraint_names_follow_the_table() -> None:
    got = _constraints()
    assert got == _EXPECTED_CONSTRAINTS, (
        f"欠け: {_EXPECTED_CONSTRAINTS - got}\n余り: {got - _EXPECTED_CONSTRAINTS}"
    )


def test_no_old_name_survives() -> None:
    """旧名（表の改名前）が索引にも制約にも残っていない。"""
    stale = {n for n in (_indexes() | _constraints())
             if n.startswith("situated_embeddings") or n.startswith("idx_se_")}
    assert not stale, f"旧名が残っている: {sorted(stale)}"
