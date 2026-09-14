"""想起の時間軸（t）と、想起で強化しないことの検査。

    score = r^(w_r) × M,  M = (w_t·t + w_e·e + w_g·a + w_p·p)/(w_t+w_e+w_g+w_p)
    t     = max(t_floor, exp(-|ref - timestamp| / tau)),  tau = HL / ln 2

**起点は書かれた時刻だけ**。強化A（想起回数で半減期を伸ばす）と強化B（使ったら若返る）は、
どちらも想起では効かせない（課題5 F節）。強化B の更新契機（フルLLM が実際に参照した MI）
の判定が未実装なので、仕組みごと後回しにしている。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import psycopg2

from familiar_agent.store.context import viewpoint_of
import psycopg2.extras
import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory():
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


def _fresh_conn():
    url = os.environ.get(
        "DATABASE_URL",
        os.environ["DATABASE_URL"],
    )
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


def test_the_time_origin_lives_on_the_facet():
    """時間の起点は出来事でなく**面**が持つ（044）。

    017 は `observations.last_recalled_at` として入れたが、044 で
    `situated_memories` へ移した。どの面を通って思い出したかで変わる量だからである
    （`設計図` [D-在席相関/V2]・`MIデータモデル` §5）。
    """
    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE column_name = 'last_recalled_at'
                  AND table_name IN ('observations', 'situated_memories')
            """)
            rows = {r["table_name"]: r for r in cur.fetchall()}
        assert "observations" not in rows, "出来事の側に残っている"
        row = rows.get("situated_memories")
        assert row is not None, "面が起点を持っていない"
        assert row["is_nullable"] == "YES", "last_recalled_at must be nullable"
        assert "timestamp" in row["data_type"], f"unexpected type: {row['data_type']}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 想起は「思い出した時」を記録する（採点の t には効かない）
# ---------------------------------------------------------------------------


def test_recall_records_when_a_memory_was_recalled(memory):
    """想起で W に載った記録は `last_recalled_at`（思い出した時）を更新する（記-a-ろ-い・2026-09-14）。

    以前は「想起では触らない」だった——当時は `last_recalled_at` が採点の t の起点でもあり、
    想起のたびに若返ると一度上がった記録が自分を押し上げ続けた（47 日前の挨拶が t=1.000）。
    いまは t の起点は作られた日（`timestamp`）だけで、`last_recalled_at` は関連想起の並びに
    しか使わないので、更新しても循環は起きない。**思い出した時の記録**として、一次想起・
    関連想起のどちらで載ったときも更新する。
    """
    memory.save("強化しない確認", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT o.id, s.last_recalled_at FROM observations o "
                "JOIN situated_memories s ON s.obs_id = o.id::text "
                "WHERE o.content = %s AND s.person_id = %s",
                ("強化しない確認", viewpoint_of(memory._person_id)),
            )
            before = cur.fetchone()

        memory.recall("強化しない確認", n=5)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_recalled_at FROM situated_memories "
                "WHERE obs_id = %s AND person_id = %s",
                (before["id"], viewpoint_of(memory._person_id)),
            )
            after = cur.fetchone()
        assert before["last_recalled_at"] is None
        assert after["last_recalled_at"] is not None, "想起しても思い出した時が記録されない"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Time-decay scoring
# ---------------------------------------------------------------------------


def test_time_decay_prioritizes_recent_over_old(memory):
    """A recently-saved memory ranks higher than one backdated 60 days, all else equal."""
    memory.save("記憶古い", kind="observation", emotion="neutral")

    conn = _fresh_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE observations SET timestamp = now() - interval '60 days' WHERE content = %s",
                ("記憶古い",),
            )
        conn.commit()
    finally:
        conn.close()

    memory.save("記憶新しい", kind="observation", emotion="neutral")

    results = memory.recall("記憶", n=10)
    scores = {r["summary"]: r["fit"] for r in results}

    assert "記憶新しい" in scores, "recent memory not found"
    assert "記憶古い" in scores, "old memory not found"
    assert scores["記憶新しい"] > scores["記憶古い"], (
        f"recent ({scores['記憶新しい']:.4f}) should exceed old ({scores['記憶古い']:.4f})"
    )


def test_recall_half_life_comes_from_the_db_not_the_env(monkeypatch, memory):
    """$HL$ は層 3 の設定値。`RECALL_HALF_LIFE_DAYS` は読まない（記-a-ろ-い）。"""
    from familiar_agent import config_overrides as co
    from familiar_agent.config import MemoryConfig

    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "14.0")
    co.clear_cache()
    assert MemoryConfig().recall_half_life_days == pytest.approx(10.0)
    assert co.save_override("MemoryConfig.recall_half_life_days", 14.0)
    co.clear_cache()
    assert MemoryConfig().recall_half_life_days == pytest.approx(14.0)
    co._delete_all()
    co.clear_cache()


def test_recall_time_floor_env_var(monkeypatch):
    """RECALL_TIME_FLOOR env var is read via MemoryConfig."""
    from familiar_agent.config import MemoryConfig

    monkeypatch.setenv("RECALL_TIME_FLOOR", "0.1")
    assert MemoryConfig().recall_time_floor == pytest.approx(0.1)
