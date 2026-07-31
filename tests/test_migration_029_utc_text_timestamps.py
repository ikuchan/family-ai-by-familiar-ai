"""Migration 029：TEXT 時刻列の既存ローカル値を UTC（aware ISO）へ寄せる。

書き込み側は `now_utc_iso()`（UTC）へ直したので、既存のローカル naive 行を同じ UTC
時計へ寄せる。tz サフィックスの無い行だけ変換し、既に UTC 化された行はスキップ（冪等）。
"""

from __future__ import annotations

import os

import importlib.util
import uuid
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.store.clock import local_tz

_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-07-23-029_utc_text_timestamps.py"


def _pg():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _run_migration(conn) -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)


def _insert_person(conn, name: str, created_at: str) -> str:
    pid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, created_at, updated_at) VALUES (%s,%s,%s,%s)",
            (pid, name, created_at, created_at),
        )
    return pid


def _created_at(conn, pid: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT created_at FROM persons WHERE id=%s", (pid,))
        return cur.fetchone()["created_at"]


def test_naive_local_value_becomes_aware_utc_same_instant():
    conn = _pg()
    naive = "2026-07-23T15:00:00.000000"  # ローカル naive（tz なし）
    pid = _insert_person(conn, f"utcmig-{uuid.uuid4()}", naive)

    _run_migration(conn)

    migrated = _created_at(conn, pid)
    # aware（+00:00）になっている。
    parsed = datetime.fromisoformat(migrated)
    assert parsed.tzinfo is not None
    # 同じ瞬間：naive をローカルと解釈した値と一致（実行マシンの tz に依存しない検証）。
    expected = datetime.fromisoformat(naive).replace(tzinfo=local_tz())
    assert parsed == expected
    conn.close()


def test_already_aware_value_is_unchanged():
    conn = _pg()
    aware = "2026-07-23T06:00:00.000000+00:00"  # 既に UTC aware
    pid = _insert_person(conn, f"utcmig-aware-{uuid.uuid4()}", aware)

    _run_migration(conn)

    assert _created_at(conn, pid) == aware  # 再変換されない（冪等）
    conn.close()


def test_missing_table_is_skipped():
    """対象表が存在しなくても落ちない。

    `_TARGETS` は「029 を書いた時点で存在した表の一覧」であって、その後の表の増減を
    織り込んでいない。表を改名すると（`memory_activation` → `memory_salience`）、この
    マイグレーションを再実行する経路が全部落ちる。無い表は飛ばす。

    効果は変わらない。新規 DB では、表を作るマイグレーションが先に走ってから 029 が
    走るので、当時存在した表はすべて揃っている。
    """
    import importlib.util

    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location("utc_text_migration_probe", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert "memory_activation" in mod._TARGETS, (
        "前提が崩れている：029 は改名前の表名を持っているはず"
    )

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('memory_activation') AS t")
        assert cur.fetchone()["t"] is None, "旧表がまだ存在する（前提が崩れている）"
    mod.upgrade(conn)      # 例外が出なければよい
    conn.close()
