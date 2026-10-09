"""075（調停の記録に、道・意味・最終の動作を足す・出-ay 段 5d・2026-10-10）。

段 4 で調停を起点ごとの道（発話・完了・情動・機器）に組み替えた。起点の欄（`origin`）だけでは完了か分からない（発話から
始まった求めの完了は「発話」と残る）ので、道（`path`）・発話の意味（`meaning`）・最終の動作（`final`）を足す。既定は空。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def test_the_path_meaning_and_final_action_are_kept():
    with psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type, column_default, is_nullable "
                "FROM information_schema.columns WHERE table_name = 'arbiter_records' "
                "AND column_name IN ('path', 'meaning', 'final')"
            )
            cols = {r["column_name"]: r for r in cur.fetchall()}
    assert set(cols) == {"path", "meaning", "final"}
    for c in cols.values():
        assert c["data_type"] == "text" and c["is_nullable"] == "NO"
        assert c["column_default"].startswith("''")


def test_it_can_be_applied_twice():
    import importlib.util
    from pathlib import Path

    path = next(Path("migration").glob("*-075_*.py"))
    spec = importlib.util.spec_from_file_location("m075", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    with psycopg2.connect(_DB_URL) as conn:
        mod.upgrade(conn)
        mod.upgrade(conn)
