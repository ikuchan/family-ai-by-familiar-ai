"""Tests for the recall score breakdown debug log（実機確認の観測点）.

実機確認では想起順が体感として妥当かを見るので、score だけでなく内訳
（r・t・a・e と現在 mood）が要る。recall はこれをログへ出していなかった。

方針は3つ。debug レベルに置く（hot path なので本番では切る）、内訳の再計算は
debug が有効なときだけ行う（無効なら素通り）、記憶内容は debug のみに載せる
（コード規約：全文や記憶内容を info 以上で出さない）。
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import patch

import psycopg2

from familiar_agent.db import get_db
from familiar_agent.mood_register import MoodPAD, save_mood
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
_CONTENT = "score breakdown log content"


def _seed() -> None:
    obs_id = str(uuid.uuid4())
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
            " person_id, activation_a0, activation_n, "
            " emotion_p, emotion_pn, emotion_a, emotion_dom) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, _CONTENT, "unknown", "conversation", "neutral", DEFAULT_PERSON_ID,
             1.0, 0, 0.8, 0.2, 0.6, 0.5),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), obs_id, DEFAULT_PERSON_ID, _VEC),
        )
    conn.close()


def _recall() -> None:
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory(person_id=DEFAULT_PERSON_ID)
    with patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]):
        mem.recall("score breakdown log content", n=5)


def _set_mood(pad: MoodPAD) -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        save_mood(conn, pad)
        conn.commit()


def test_debug_log_carries_score_breakdown(caplog) -> None:
    """debug で score の内訳（r・t・a・e）と現在 mood が出る。"""
    _seed()
    _set_mood(MoodPAD(0.7, 0.3, 0.5, 0.5))
    with caplog.at_level(logging.DEBUG, logger="familiar_agent.tools.memory"):
        _recall()

    lines = [r.getMessage() for r in caplog.records]
    breakdown = [ln for ln in lines if "recall score" in ln]
    assert breakdown, f"内訳ログが出ていない: {lines}"
    joined = "\n".join(breakdown)
    for field in ("r=", "t=", "a=", "e=", "mood="):
        assert field in joined, f"{field} が内訳ログに無い:\n{joined}"


def test_no_breakdown_log_below_debug(caplog) -> None:
    """info 以上では内訳を出さない（hot path・記憶内容を含むため）。"""
    _seed()
    _set_mood(MoodPAD(0.7, 0.3, 0.5, 0.5))
    with caplog.at_level(logging.INFO, logger="familiar_agent.tools.memory"):
        _recall()

    assert not [r for r in caplog.records if "recall score" in r.getMessage()]
