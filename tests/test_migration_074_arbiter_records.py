"""074（調停ごとの記録 `arbiter_records`・出-ay 段 3・2026-10-08）。

正解（`JEV_正解.md`）と突き合わせて Jev のプロンプトを直すため、調停のたびに Jev に渡したそのままの文・問い・答え・
結末を残す（本人の決定：本番 DB に残す・消さない）。W は DEBUG のログにしか無く、後から組み直すと想起の点数が変わって
同じにならない。記-o（W に何が載っていたか）もこれで見る。
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _columns() -> dict[str, str]:
    with psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'arbiter_records'"
            )
            return {r["column_name"]: r["data_type"] for r in cur.fetchall()}


def test_the_table_has_what_jev_saw_and_said():
    cols = _columns()
    # 075 で道・意味・最終の動作を足した（出-ay 段 5d）。ここでは 074 が作った列を見る。
    assert {k: v for k, v in cols.items() if k not in ("path", "meaning", "final")} == {
        "id": "bigint",
        "at": "timestamp with time zone",
        "origin": "text",
        "utterance": "text",
        "state": "text",
        "questions": "jsonb",
        "answer": "jsonb",
        "outcome": "text",
    }


def test_records_are_found_by_time():
    with psycopg2.connect(_DB_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'arbiter_records' AND indexdef LIKE '%(at)%'"
        )
        assert cur.fetchone() is not None
