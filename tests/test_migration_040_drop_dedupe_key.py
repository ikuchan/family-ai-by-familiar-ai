"""040（`memory_events.dedupe_key` の撤去）が効いていることを確かめる。

重複防止は2つあった。鍵による突き合わせ（`memory_events.dedupe_key`・キューへ積む段）と、
時間窓（`observations` への書き込み時・30秒・内容と kind の一致）である。

鍵が実際に弾いていたのは同じターン内の再書き込みで、いずれも30秒に収まる。日次要約に
ついては元から効いていなかった（鍵の digest が content の sha1 で、要約は毎回中身が違う
ため一致しない）。よって時間窓へ一本化する。
"""

from __future__ import annotations

import inspect
import os

import psycopg2

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    return psycopg2.connect(_DB_URL)


def test_dedupe_key_column_is_gone() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'memory_events' AND table_schema = 'public'"
            )
            cols = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "dedupe_key" not in cols
    assert {"event_id", "created_at", "event_type", "payload_json", "person_id"} <= cols


def test_dedupe_index_is_gone() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'public' AND tablename = 'memory_events'"
            )
            names = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "idx_memory_events_dedupe" not in names


def test_append_memory_event_no_longer_takes_dedupe_key() -> None:
    """口からも消えていること（列だけ落として引数が残ると、黙って無視される）。"""
    from familiar_agent.tools.memory import ObservationMemory

    sig = inspect.signature(ObservationMemory.append_memory_event)
    assert "dedupe_key" not in sig.parameters
    for name in ("save_async_with_id", "save_with_id"):
        assert "dedupe_key" not in inspect.signature(getattr(ObservationMemory, name)).parameters, name
