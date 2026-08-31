"""041（`unfinished_business` の撤去）が効いていることを確かめる。

開いた意図は O の記録（`direction="求め"`）が担っており、この表は二重の実装だった。
書き手は `heartbeat._persist_remainder` の1箇所で、それを呼ぶ `apply_status` は
環-c（旧 `run()` の撤去・2026-07-29）で呼び出し側を失っていた。読み手は本番0件。
"""

from __future__ import annotations

import os

import psycopg2

_DB_URL = os.environ["DATABASE_URL"]


def _tables() -> set[str]:
    conn = psycopg2.connect(_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def test_unfinished_business_table_is_gone() -> None:
    assert "unfinished_business" not in _tables()


def test_memory_no_longer_exposes_unfinished_business() -> None:
    """表を落とすだけでは足りない。読み書きの口も消えていること。"""
    from familiar_agent.tools.memory import ObservationMemory

    for name in ("open_unfinished_business",
                 "list_unfinished_business",
                 "list_unfinished_business_async"):
        assert not hasattr(ObservationMemory, name), name


def test_heartbeat_runtime_is_gone() -> None:
    """継続制御の器ごと落ちていること（本番呼び出しは全メソッド0件だった）。"""
    import importlib

    try:
        importlib.import_module("familiar_agent.heartbeat")
    except ModuleNotFoundError:
        return
    raise AssertionError("familiar_agent.heartbeat がまだ import できる")
