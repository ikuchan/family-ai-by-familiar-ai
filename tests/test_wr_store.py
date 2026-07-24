"""WRDB スライス1：ターンの想起 MI 集合を WR として記録する（拡散は未接続・挙動不変）。"""

from __future__ import annotations

import os

import psycopg2

from familiar_agent.wr_store import combine_wr_ids, load_wr_items, save_wr


# ── combine_wr_ids（W 想起 MI ＋ そのターンの新記憶・順序保存で重複除去・純関数） ──

def test_combine_recalled_and_new_ids():
    memories = [{"memory_id": "w1"}, {"memory_id": "w2"}]
    out = combine_wr_ids(memories, ["obs1", "conv1"])
    assert out == ["w1", "w2", "obs1", "conv1"]  # W ＋ 新記憶が1つの WR に共起


def test_combine_dedups_and_skips_none():
    memories = [{"memory_id": "w1"}, {"memory_id": None}, {"memory_id": "w1"}]
    out = combine_wr_ids(memories, [None, "conv1", "w1"])
    assert out == ["w1", "conv1"]  # 重複・None を除去


def test_combine_empty():
    assert combine_wr_ids(None, None) == []
    assert combine_wr_ids([], []) == []


def _conn():
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    c.autocommit = True
    return c


def test_save_wr_roundtrip():
    conn = _conn()
    wr_id = save_wr(conn, ["mi-a", "mi-b", "mi-c"])
    assert wr_id is not None
    assert load_wr_items(conn, wr_id) == ["mi-a", "mi-b", "mi-c"]
    conn.close()


def test_save_wr_empty_records_nothing():
    conn = _conn()
    assert save_wr(conn, []) is None
    assert save_wr(conn, [None, "", None]) is None  # 空要素だけ
    conn.close()


def test_cooccurrence_countable_across_wrs():
    """別行構造で「共起（共有 mi_id）件数」が GROUP BY で数えられる（(A) 共起辺の土台）。"""
    conn = _conn()
    wr1 = save_wr(conn, ["x1", "x2", "x3"])
    wr2 = save_wr(conn, ["x2", "x3", "x9"])  # x2,x3 を wr1 と共有＝共起2
    with conn.cursor() as cur:
        cur.execute(
            "SELECT wr_id, COUNT(*) FROM wr_record_items "
            "WHERE mi_id IN ('x1','x2','x3') AND wr_id <> %s "
            "GROUP BY wr_id",
            (wr1,),
        )
        rows = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    assert rows.get(wr2) == 2  # wr2 は wr1 と2件共起
