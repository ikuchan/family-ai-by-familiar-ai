"""Tests for wiring the e 軸 into recall（Phase 2 スライス3・気分一致想起）.

想起スコアがハイブリッド合成になり、e＝**今の気分と観測 PAD の距離**が加算部の
一項として効く。同じコサイン・同じ時刻・同じ 根づき の2件を置き、感情だけを
変えたとき、気分に近いほうが上位へ来ることを見る。

あわせてデッドロックの反証を置く：mood の読み出しは `db.lock` を取るので、
recall の DB ロック内から呼ぶと停止する（平均中心化 C2 と同型）。recall が
有限時間で返ることを時間で押さえる。
"""

from __future__ import annotations

import os

import threading
import time
import uuid
from unittest.mock import patch

import psycopg2

from familiar_agent.db import get_db
from familiar_agent.mood_register import MoodPAD, save_mood
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"

_CONTENT_GLAD = "emotion axis glad memory"
_CONTENT_GLUM = "emotion axis glum memory"


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory(person_id=DEFAULT_PERSON_ID)


def _seed(content: str, pad: tuple[float, float, float, float]) -> None:
    """コサイン・時刻・根づき を揃え、PAD だけが違う観測を置く。"""
    obs_id = str(uuid.uuid4())
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
            " person_id, groundedness_g0, groundedness_n, "
            " emotion_p, emotion_pn, emotion_a, emotion_dom) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, content, "unknown", "conversation", "neutral", DEFAULT_PERSON_ID,
             1.0, 0, *pad),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), obs_id, DEFAULT_PERSON_ID, _VEC),
        )
    conn.close()


def _set_mood(pad: MoodPAD) -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        save_mood(conn, pad)
        conn.commit()


def _scores() -> dict[str, float]:
    with patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]):
        results = _mem().recall("emotion axis memory", n=50)
    by_content = {r["summary"]: r["fit"] for r in results}
    assert _CONTENT_GLAD in by_content and _CONTENT_GLUM in by_content, (
        "seeded observations were not recalled"
    )
    return by_content


def test_recall_favours_memories_matching_the_current_mood() -> None:
    """明るい気分では明るい記憶が、沈んだ気分では沈んだ記憶が上位へ来る。"""
    _seed(_CONTENT_GLAD, (0.9, 0.1, 0.6, 0.6))
    _seed(_CONTENT_GLUM, (0.1, 0.9, 0.4, 0.4))

    _set_mood(MoodPAD(0.9, 0.1, 0.6, 0.6))
    glad_mood = _scores()
    assert glad_mood[_CONTENT_GLAD] > glad_mood[_CONTENT_GLUM]

    # 反証側：気分を逆へ振ると順位が入れ替わる（たまたま片方が強いのではない）
    _set_mood(MoodPAD(0.1, 0.9, 0.4, 0.4))
    glum_mood = _scores()
    assert glum_mood[_CONTENT_GLUM] > glum_mood[_CONTENT_GLAD]


def test_recall_returns_without_deadlock() -> None:
    """recall が有限時間で返る（mood 読みが DB ロック内なら停止する）。"""
    _seed(_CONTENT_GLAD, (0.9, 0.1, 0.6, 0.6))
    _seed(_CONTENT_GLUM, (0.1, 0.9, 0.4, 0.4))
    _set_mood(MoodPAD(0.6, 0.4, 0.5, 0.5))

    done = threading.Event()

    def _run() -> None:
        with patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]):
            _mem().recall("emotion axis memory", n=5)
        done.set()

    worker = threading.Thread(target=_run, daemon=True)
    started = time.monotonic()
    worker.start()
    assert done.wait(timeout=30.0), "recall がデッドロックした（mood 読みがロック内）"
    assert time.monotonic() - started < 30.0
