"""055：`situated_memories.relation_key` の既定値を `present` へ揃える。

**047 の原本は既定値を `presence` のままにしていた。** 2026-08-21 のダンプの
`situated_memories.relation_key` は `DEFAULT 'presence'` で、復元した 047 が `present` へ
変えたため、本番とテストでスキーマが食い違っていた（2026-09-02 に機械差分で見つけた・
`復旧記録` v0.18）。

役割名は 047 以降すべて `present` に揃っている。実データに `presence` の行は 1 件も無く、
全行が明示的に値を入れているので既定値を使う経路は無い。**既定値だけが古い名前で残って
いた**ので、そちらを実物へ合わせる（案B）。

テスト DB には復元版 047 が走っていて既定値はもう `present` なので、**旧い既定値を置き直して
から**マイグレーションを流す。そうしないと、効いているのか初めからそうなのかを見分けられない。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-09-02-055_relation_key_default.py"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _default_of(column: str = "relation_key") -> str | None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_name = 'situated_memories' AND column_name = %s",
                (column,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return None if row is None else row["column_default"]


def _set_default(value: str) -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE situated_memories ALTER COLUMN relation_key SET DEFAULT '{value}'"
            )
    finally:
        conn.close()


def _run_migration() -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    conn = psycopg2.connect(_DB_URL)
    try:
        mod.upgrade(conn)
        conn.commit()
    finally:
        conn.close()


def test_migration_replaces_the_old_default() -> None:
    """旧い既定値を置いてから流すと、新しい名前へ変わる。"""
    try:
        _set_default("presence")
        assert _default_of() == "'presence'::text", "前提：旧い既定値を置けている"
        _run_migration()
        assert _default_of() == "'present'::text"
    finally:
        _set_default("present")


def test_migration_is_idempotent() -> None:
    """すでに新しい名前でも、二度流して壊れない。"""
    _run_migration()
    _run_migration()
    assert _default_of() == "'present'::text"


def test_the_column_is_still_not_null() -> None:
    """既定値を替えただけで、NOT NULL は動かしていない（反証側）。"""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_name='situated_memories' AND column_name='relation_key'"
            )
            assert cur.fetchone()["is_nullable"] == "NO"
    finally:
        conn.close()
