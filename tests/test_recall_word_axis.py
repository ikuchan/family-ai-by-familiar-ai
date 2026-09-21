"""語の軸と底上げが想起まで効くか（記-k・2026-09-21）。実 DB を使う。

実機 15:58：5 分前に自分が話した「…キャンプの準備…本を更新…」を、5 分後に「さっき本の話
したよね？覚えてる？」と聞いても「思い出せない」と答えた。その記録は**候補には入っていた**
（時間の軸で 32 位）が、採点で 11 位となり 7 件に届かなかった。**取りこぼしは候補集めではなく
採点にあった。**

ここで見るのは、語の軸を足して軸ごとの順位で底上げすると、その記録が 7 件に入ること。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import psycopg2

from familiar_agent.config import RecallWeights
from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DIM = 1024
#: 発話で起こる想起の重み（`課題5_パラメータ仮案` の採用値）。
_W = RecallWeights(1.50, 0.93, 0.99, 1.43, 1.49)
_QUERY = "さっき本の話したよね？覚えてる？"
#: 探している記録。語「本」を含み、5 分前に書かれ、問いとは似ていない。
_TARGET = "（誰も見えないあいだに聞いた）今日は朝ご飯を食べて、キャンプの準備について相談した"


def _conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _memory():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory().for_person(AGENT_SELF_ID)


def _unit(v: "np.ndarray") -> "np.ndarray":
    return v / (float(np.linalg.norm(v)) or 1.0)


def _vec_sql(v: "np.ndarray") -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _plant(body: str, vec: "np.ndarray", when: "datetime") -> str:
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (obs_id, body, when, "会話", "conversation", "neutral"),
            )
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, vector, relation_key, content, last_recalled_at) "
                "VALUES (%s,%s,%s,%s::vector,%s,%s,%s)",
                (
                    str(uuid.uuid4()),
                    obs_id,
                    AGENT_SELF_ID,
                    _vec_sql(vec),
                    "actor",
                    body,
                    when,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return obs_id


def _at_cosine(q_hat: "np.ndarray", cos: float, rng) -> "np.ndarray":
    """問いとのコサインをちょうど `cos` にしたベクトルを作る。"""
    u = rng.normal(size=len(q_hat)).astype(np.float32)
    u = _unit(u - float(np.dot(u, q_hat)) * q_hat)  # 問いと直交する向き
    return _unit(cos * q_hat + float(np.sqrt(max(0.0, 1.0 - cos * cos))) * u)


def _plant_the_scene(mem) -> str:
    """実機の比率をそのまま作る。

    実機では、探していた記録の類似は 0.235、7 件目の境目は 0.40 だった。問いに少し似た
    記録 8 件（語を含まない・1 日前）が 7 席を埋め、探している記録（5 分前・語を含む・
    似ていない）は採点だけでは 9 番目になり、床 0.05 にも届かない。
    """
    from familiar_agent.store.situated import _situated_vector

    q_vec = np.array(mem._embedder.encode_query([_QUERY])[0], dtype=np.float32)
    q_hat = _unit(_situated_vector(q_vec, mem._situated._embedding_mu()))
    now = datetime.now(timezone.utc)
    rng = np.random.default_rng(20260921)
    for i in range(8):
        _plant(
            f"きのう見た番組の感想を言った（{i}）",
            _at_cosine(q_hat, 0.30, rng),
            now - timedelta(days=1),
        )
    return _plant(
        _TARGET + "あと、午前中は本を更新していた",
        _at_cosine(q_hat, 0.235, rng),
        now - timedelta(minutes=5),
    )


def test_the_five_minute_old_record_with_the_word_takes_a_seat():
    mem = _memory()
    target = _plant_the_scene(mem)
    got = mem.recall(_QUERY, n=7, min_score=0.05, weights=_W)
    assert target in [r["memory_id"] for r in got], "語を含む 5 分前の記録が 7 件に入らない"


def test_a_question_without_content_words_is_unchanged():
    """中身の語が無い問いでは語の軸を引かず、いままでの並びのまま。"""
    mem = _memory()
    _plant_the_scene(mem)
    got = mem.recall("覚えてる？", n=7, min_score=0.05, weights=_W)
    assert got, "語が無い問いで想起が空になってはいけない"
