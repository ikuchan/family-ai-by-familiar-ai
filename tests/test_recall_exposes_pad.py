"""Tests for recall が PAD と 根づきの重みを露出する（mood-b）。

nudge（mood-c）の入力に、W の各記憶の PAD（MoodPAD）と 根づきの重み
（_derive_groundedness(a0,n)）が要る。recall の返り dict にこの2フィールドを足す
（追加のみ・挙動不変）。
"""

from __future__ import annotations

import os

import uuid
from unittest.mock import patch

import psycopg2

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel, _derive_groundedness
from familiar_agent.mood_register import MoodPAD
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID


_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory(person_id=DEFAULT_PERSON_ID)


def _seed_one(obs_id: str) -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id, "
            " groundedness_g0, emotion_p, emotion_pn, emotion_a, emotion_dom) "
            "VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (obs_id, "pad recall content", "unknown", "conversation", "happy", DEFAULT_PERSON_ID,
             0.75, 0.8, 0.15, 0.55, 0.6),
        )
        cur.execute(
            "INSERT INTO situated_memories (id, obs_id, person_id, vector) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), obs_id, DEFAULT_PERSON_ID, _VEC),
        )
    conn.close()


def _recall_one(mem: ObservationMemory) -> dict:
    with patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]):
        results = mem.recall("pad recall content", n=5)
    hit = [r for r in results if r["summary"] == "pad recall content"]
    assert hit, "seeded observation was not recalled"
    return hit[0]


# ── 1. PAD 露出 ─────────────────────────────────────────────────────────────
def test_recall_exposes_emotion_pad() -> None:
    obs_id = str(uuid.uuid4())
    _seed_one(obs_id)
    m = _recall_one(_mem())
    assert m["emotion_pad"] == MoodPAD(0.8, 0.15, 0.55, 0.6)


# ── 2. 根づきの重み露出（n=0 なら a0） ───────────────────────────────────
def test_recall_exposes_activation_weight() -> None:
    obs_id = str(uuid.uuid4())
    _seed_one(obs_id)
    m = _recall_one(_mem())
    assert m["groundedness"] == _derive_groundedness(0.75, 0)


# ── 3. 既存キーは不変（反証） ───────────────────────────────────────────────
def test_recall_keeps_existing_keys() -> None:
    obs_id = str(uuid.uuid4())
    _seed_one(obs_id)
    m = _recall_one(_mem())
    for key in ("memory_id", "summary", "emotion", "fit", "kind"):
        assert key in m
    assert m["emotion"] == "happy"  # 文字列 emotion は従来どおり
