"""PostgreSQL migration runner.

Migration files receive a psycopg2 connection and implement:
    def upgrade(conn: psycopg2.extensions.connection) -> None: ...

Each file uses `with conn.cursor() as cur: cur.execute(...)`.
The runner handles transactions, applied-tracking, and ordering.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Callable

import psycopg2.extensions

logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE = "schema_migrations"


def default_migration_dir() -> Path:
    env = os.getenv("FAMILIAR_AI_MIGRATION_DIR")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "migration"
        if candidate.exists():
            return candidate
    return Path("migration")


def _ensure_migrations_table(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_MIGRATIONS_TABLE} (
                id          TEXT PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def _applied_ids(conn: psycopg2.extensions.connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {_MIGRATIONS_TABLE}")
        return {row[0] for row in cur.fetchall()}


def _load_module(path: Path) -> ModuleType:
    name = "migration_" + re.sub(r"[^a-zA-Z0-9_]", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load migration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _migration_files(mig_dir: Path) -> list[Path]:
    """適用対象のスクリプトを辞書順で返す。`_` 始まりは補助なので除く。

    列挙の規則をここ1箇所に置く。`apply_migrations` と `pending_migration_ids` が
    別々に持つと、片方だけ直したときに「適用される一覧」と「保留として見せる一覧」が
    食い違う。
    """
    if not mig_dir.exists():
        logger.warning("Migration directory not found: %s", mig_dir)
        return []
    files = sorted(p for p in mig_dir.glob("*.py") if not p.name.startswith("_"))
    if not files:
        logger.warning("No migration scripts in: %s", mig_dir)
    return files


def pending_migration_ids(
    conn: psycopg2.extensions.connection,
    migration_dir: Path | None = None,
) -> list[str]:
    """まだ適用していないスクリプトの id を辞書順で返す（適用はしない）。

    自動適用を留めたとき、何が保留されているかを見せるのに使う。
    """
    mig_dir = migration_dir or default_migration_dir()
    _ensure_migrations_table(conn)
    applied = _applied_ids(conn)
    return [p.stem for p in _migration_files(mig_dir) if p.stem not in applied]


def apply_migrations(
    conn: psycopg2.extensions.connection,
    migration_dir: Path | None = None,
) -> int:
    """Apply pending migration scripts in lexical order. Returns count applied."""
    mig_dir = migration_dir or default_migration_dir()
    _ensure_migrations_table(conn)

    files = _migration_files(mig_dir)
    if not files:
        return 0

    applied = _applied_ids(conn)
    count = 0

    for path in files:
        mid = path.stem
        if mid in applied:
            continue
        module = _load_module(path)
        upgrade: Callable | None = getattr(module, "upgrade", None)
        if not callable(upgrade):
            raise RuntimeError(f"upgrade(conn) missing in {path}")
        try:
            upgrade(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {_MIGRATIONS_TABLE} (id) VALUES (%s)", (mid,)
                )
            conn.commit()
            applied.add(mid)
            count += 1
            logger.info("Applied migration: %s", mid)
        except Exception:
            conn.rollback()
            logger.exception("Failed migration: %s", mid)
            raise

    return count
