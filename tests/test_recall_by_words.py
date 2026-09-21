"""語で引く列（記-k・2026-09-21）。実 DB を使う。

ベクトルの列だけでは、問いの中身の語を含む記録が落ちる（実機 15:58・探していた発話が 822 位）。
そこで語の列を独立に作る。ここで見るのは**語で引く口**（`observations.by_words`）だけで、
並べ方（順位で混ぜる）は `core/keyword_rules` が持つ。
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import numpy as np
import psycopg2

from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DIM = 1024


def _conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _vec_sql(seed: int) -> str:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_DIM).astype(np.float32)
    v = v / (float(np.linalg.norm(v)) or 1.0)
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _store():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory().for_person(AGENT_SELF_ID)._observations


def _plant(body: str, seed: int = 7) -> str:
    """1 つの記録を、自分の面で植える。"""
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,now(),%s,%s,%s)",
                (obs_id, body, "会話", "conversation", "neutral"),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key, content) "
                "VALUES (%s,%s,%s,%s::vector,%s,%s)",
                (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _vec_sql(seed), "actor", body),
            )
        conn.commit()
    finally:
        conn.close()
    return obs_id


def test_a_record_with_the_word_comes_back():
    _plant("今日はキャンプの準備をして、本を更新した")
    _plant("明日の天気は晴れ")
    got = _store().by_words(["キャンプ"], 10)
    assert len(got) == 1
    assert "キャンプ" in got[0]["content"]


def test_several_words_are_any_of_them():
    _plant("キャンプの準備をした")
    _plant("写真を撮った")
    _plant("まったく関係のない話")
    got = _store().by_words(["キャンプ", "写真"], 10)
    assert len(got) == 2


def test_the_newest_come_first():
    _plant("キャンプ その1")
    _plant("キャンプ その2")
    got = _store().by_words(["キャンプ"], 10)
    assert got[0]["content"].endswith("その2"), [r["content"] for r in got]


def test_no_words_means_no_rows():
    _plant("キャンプの準備をした")
    assert _store().by_words([], 10) == []


def test_excluded_ids_do_not_come_back():
    oid = _plant("キャンプの準備をした")
    assert _store().by_words(["キャンプ"], 10, exclude_ids=[oid]) == []


def test_the_rows_look_like_the_vector_ones():
    """後ろの採点が同じ形を受け取れること（`by_vector` と同じ列）。"""
    _plant("キャンプの準備をした")
    row = _store().by_words(["キャンプ"], 10)[0]
    for key in ("id", "timestamp", "content", "groundedness_g0", "groundedness_n", "facet_id"):
        assert key in row, key


# ── 語がどれだけありふれているかを数える ─────────────────────────────────────


def test_it_counts_how_many_records_hold_the_word():
    """語ごとの当たり数と、母数（自分の面の総数）を一度に返す。"""
    _plant("今日はキャンプの準備をして、本を更新した")
    _plant("キャンプ用の鍋を買った")
    _plant("明日の天気は晴れ")
    counts, total = _store().word_counts(["キャンプ", "天気"])
    assert counts["キャンプ"] == 2
    assert counts["天気"] == 1
    assert total == 3


def test_a_word_nobody_said_counts_zero():
    _plant("明日の天気は晴れ")
    counts, total = _store().word_counts(["散歩"])
    assert counts["散歩"] == 0
    assert total == 1


def test_no_words_no_counts():
    _plant("明日の天気は晴れ")
    assert _store().word_counts([]) == ({}, 1)
