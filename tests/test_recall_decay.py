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
# 想起では強化しない（強化B は仕組みごと後回し）
# ---------------------------------------------------------------------------


def test_recall_never_reinforces(memory):
    """想起は時間の起点（`last_recalled_at`）を触らない。

    更新すべきは「フルLLM が実際に参照した MI」だけ（課題5 F節・強化B「想起では触らない」）
    だが、その判定は未実装なので仕組みごと後回しにした。想起しただけで若返らせると、
    t の起点が毎回 now に戻り、**一度上がった記録が自分を押し上げ続ける**。実機では
    47日前の挨拶が t=1.000 で居座り、5秒前の自分の発話を W から押し出した
    （「おかえりなさい」を2回言った）。
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
        assert after["last_recalled_at"] == before["last_recalled_at"]
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
                "UPDATE observations SET timestamp = now() - interval '60 days' "
                "WHERE content = %s",
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


def test_recall_half_life_env_var(monkeypatch, memory):
    """RECALL_HALF_LIFE_DAYS env var is read via MemoryConfig."""
    from familiar_agent.config import MemoryConfig
    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "14.0")
    assert MemoryConfig().recall_half_life_days == pytest.approx(14.0)


def test_recall_time_floor_env_var(monkeypatch):
    """RECALL_TIME_FLOOR env var is read via MemoryConfig."""
    from familiar_agent.config import MemoryConfig
    monkeypatch.setenv("RECALL_TIME_FLOOR", "0.1")
    assert MemoryConfig().recall_time_floor == pytest.approx(0.1)
