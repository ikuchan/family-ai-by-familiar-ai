"""078（記録の種類 `kind` の列と索引を落とし、向き `direction` に索引を足す・記-u 段 4・2026-10-27・本人の決定）。

「何の記録か」は向きだけで持つ（記-u・案ウ）。種類にしか無かった情報は 077 で向きへ移し、種類で引いていた
読み手は段 2 で向きへ移した。書く口は段 3 で OIF に一本化した。向きで引く読み手（記憶の木・最終版・畳み込みの
材料）のために、向きに索引を足す。
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_PATH = (
    Path(__file__).resolve().parents[1] / "migration" / "2026-10-27-078_drop_observation_kind.py"
)


def _conn():
    return psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _schema(cur) -> "tuple[bool, set[str]]":
    """(種類の列があるか, observations の索引の名前)。"""
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'observations' AND column_name = 'kind'"
    )
    has_kind = cur.fetchone() is not None
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'observations'")
    return has_kind, {r["indexname"] for r in cur.fetchall()}


def test_the_kind_column_is_gone_and_direction_is_indexed():
    conn = _conn()
    try:
        with conn.cursor() as cur:
            has_kind, indexes = _schema(cur)
            cur.execute(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'observations' AND indexname = 'idx_obs_direction'"
            )
            row = cur.fetchone()
        assert not has_kind
        assert "idx_obs_kind" not in indexes
        assert row is not None and "(direction)" in row["indexdef"]
    finally:
        conn.rollback()
        conn.close()


def test_upgrade_drops_kind_and_is_idempotent():
    """1 つのトランザクションの中で、落とす前の形に戻してから当て、見て、巻き戻す。"""
    from familiar_agent.db_migrations import _load_module

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE observations "
                "ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'observation'"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_kind ON observations(kind)")
            cur.execute("DROP INDEX IF EXISTS idx_obs_direction")
            has_kind, indexes = _schema(cur)
            # **反証の側**：当てる前は、列も索引も落とす前の形である。
            assert has_kind and "idx_obs_kind" in indexes and "idx_obs_direction" not in indexes
        mod = _load_module(_PATH)
        mod.upgrade(conn)
        mod.upgrade(conn)  # 冪等：2 度目は何もしない
        with conn.cursor() as cur:
            has_kind, indexes = _schema(cur)
        assert not has_kind
        assert "idx_obs_kind" not in indexes and "idx_obs_direction" in indexes
    finally:
        conn.rollback()
        conn.close()
