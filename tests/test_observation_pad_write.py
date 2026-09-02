"""Tests for 書き込み経路の PAD 配管（W2b-1・外部挙動不変）。

`save` 系が `emotion_pad`（MoodPAD）を受けて観測行の PAD 列へ保存できるようにする。
PAD は payload（JSON・to_json_dict/from_json_dict）経由で遅延マテリアライズも通る。
未指定は中立0.5（列既定と同値）。呼び出し側の PAD 引き渡しは W2b-2（この段では未接続）。
"""

from __future__ import annotations

import os

import uuid
from unittest.mock import patch

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.mood_register import MoodPAD


_DB_URL = os.environ["DATABASE_URL"]

_FIXED_VEC = np.zeros(1024, dtype=np.float32)
_FIXED_VEC[0] = 1.0


def _pg_conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _pad_of(obs_id: str) -> tuple[float, float, float, float]:
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT emotion_p, emotion_pn, emotion_a, emotion_dom FROM observations WHERE id = %s",
            (obs_id,),
        )
        r = cur.fetchone()
    conn.close()
    return (r["emotion_p"], r["emotion_pn"], r["emotion_a"], r["emotion_dom"])


# ── 1. PAD 付き保存が PAD 列へ入る ─────────────────────────────────────────
def test_save_with_pad_writes_columns() -> None:
    mem = _mem()
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED_VEC]):
        obs_id, _ = mem.save_with_id(
            "pad write test " + uuid.uuid4().hex,
            emotion_pad=MoodPAD(0.8, 0.15, 0.55, 0.6),
            materialize_now=True,
        )
    assert obs_id is not None
    assert _pad_of(obs_id) == (0.8, 0.15, 0.55, 0.6)


# ── 2. PAD 無し保存は中立0.5（外部挙動不変） ───────────────────────────────
def test_save_without_pad_leaves_it_unmeasured() -> None:
    """PAD を渡さなければ**未測定**として書く（050）。A だけは機械値なので入る。

    050 の前は中立（0.5×4）で埋めていた。埋めると「測ったのか埋めたのか」が後から
    見分けられず、感情軸の母集合が一点に潰れる（本番 6433 行のうち 2941 行がそうなって
    いた）。REST 内省が埋め直す余地も消える。
    """
    mem = _mem()
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED_VEC]):
        obs_id, _ = mem.save_with_id(
            "no pad test " + uuid.uuid4().hex,
            materialize_now=True,
        )
    assert obs_id is not None
    p, pn, a, dom = _pad_of(obs_id)
    assert (p, pn, dom) == (None, None, None)
    assert a is not None, "A（高ぶり）は機械値なので常に入る"


# ── 3. 遅延 payload 往復（json.loads 後の dict を from_json_dict で戻す） ────
def test_materialize_from_payload_dict_round_trips_pad() -> None:
    mem = _mem()
    obs_id = str(uuid.uuid4())
    payload = {
        "content": "deferred pad " + uuid.uuid4().hex,
        "direction": "会話",
        "kind": "conversation",
        "emotion": "happy",
        "emotion_pad": MoodPAD(0.7, 0.2, 0.35, 0.5).to_json_dict(),
    }
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED_VEC]):
        stored_id = mem._observations.materialize_save_event(obs_id, payload)
    # 返るのは「この内容を保持する行の id」（重複スキップなら既存行の id）。
    assert stored_id == obs_id
    assert _pad_of(obs_id) == (0.7, 0.2, 0.35, 0.5)
