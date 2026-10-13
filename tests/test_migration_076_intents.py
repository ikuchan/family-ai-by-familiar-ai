"""076（意図の台帳の記録 `intents`・出-cc・2026-10-13・本人の決定）。

意図（調べもの・道具・主LLM の答え）ごとに uuid を振り、完了は uuid で引く。引き当てはメモリ上の台帳で、ここは
あとから追うための記録（投げた・終わった・どう終わったか）。結果の中身は持たない（O にあり、家族の話も含む）。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]

_EXPECTED = {
    "id": "uuid",
    "request_id": "text",
    "origin": "text",
    "action": "text",
    "query": "text",
    "state": "text",
    "failed": "boolean",
    "result_chars": "integer",
    "opened_at": "timestamp with time zone",
    "ended_at": "timestamp with time zone",
}


def test_the_table_has_the_intent_columns():
    with psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'intents'"
            )
            cols = {r["column_name"]: r["data_type"] for r in cur.fetchall()}
            cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'intents'")
            indexes = " ".join(r["indexdef"] for r in cur.fetchall())
    assert cols == _EXPECTED
    assert "(request_id)" in indexes and "(opened_at)" in indexes


def test_it_can_be_applied_twice():
    import importlib.util
    from pathlib import Path

    path = next(Path("migration").glob("*-076_*.py"))
    spec = importlib.util.spec_from_file_location("m076", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    with psycopg2.connect(_DB_URL) as conn:
        mod.upgrade(conn)
        mod.upgrade(conn)
