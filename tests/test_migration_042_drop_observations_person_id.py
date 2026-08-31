"""042（`observations.person_id` の撤去）が効いていることを確かめる。

所有者絞りは実データで機能していなかった。2026-08-03 時点の 5080 行のうち
4904 行（96.5%）が `default` に潰れており、家族4人のうち2人は所有行を1件も
持たない。人ごとの区別は situated 側（`s.person_id`）が担っている。

C-1 はフォールバック二関数に「situated 行を持たない観測を拾う」役目を残したが、
その母集合は現在 0 行である（生存する観測 2672 件すべてが situated 行を持つ）。
役目が消えたので、フォールバックからも所有者絞りを外す。

重複判定の30秒窓だけは絞りを保つ。ただし所有者ではなく書き手で絞る
（別の人が同じ言葉を30秒以内に言ったものは重複ではない）。それは
`test_bug1_utterance_dedup.py` で確かめる。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID
from familiar_agent.store import clock
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_DB_URL = os.environ["DATABASE_URL"]

_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=clock.local_tz())


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _insert(cur, obs_id: str, content: str, kind: str, writer_id: str, ts: datetime) -> None:
    """所有者列を書かずに観測を1件植える（042 後の列構成で書けることも兼ねて確かめる）。"""
    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion, writer_id, subject_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", writer_id, writer_id),
    )


def test_observations_has_no_person_id_column() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'observations'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "person_id" not in columns, sorted(columns)


def test_read_observations_by_kind_does_not_filter_by_owner() -> None:
    """旧仕様（所有者で絞る）を置き換えた。書き手が違っても同じ kind なら両方返る。

    042 の前はここが1件だった。所有者絞りが `default` に潰れていた以上、
    絞りは人を分けておらず、分けているように見えるのはテストの中だけだった。
    """
    tag = uuid.uuid4().hex[:8]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _insert(cur, f"self-{tag}", f"agent curiosity {tag}", "curiosity", AGENT_SELF_ID, _NOW)
            _insert(cur, f"user-{tag}", f"user curiosity {tag}", "curiosity", DEFAULT_PERSON_ID,
                    _NOW + timedelta(seconds=1))
    finally:
        conn.close()

    # 引数から person_id が消えたことと、絞りが効かなくなったことの両方を見る。
    # 植えた行の所有者は `default` なので、AGENT_SELF の文脈で引いて両方返れば
    # 所有者絞りは残っていない。
    rows = _mem().for_person(AGENT_SELF_ID)._observations._read_observations_by_kind(
        "curiosity", 10, ("content", "timestamp")
    )
    contents = {r["content"] for r in rows}

    assert f"agent curiosity {tag}" in contents
    assert f"user curiosity {tag}" in contents


def test_recency_fallback_does_not_filter_by_owner() -> None:
    tag = uuid.uuid4().hex[:8]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _insert(cur, f"r-self-{tag}", f"recency self {tag}", "observation", AGENT_SELF_ID, _NOW)
            _insert(cur, f"r-user-{tag}", f"recency user {tag}", "observation", DEFAULT_PERSON_ID,
                    _NOW + timedelta(seconds=1))
    finally:
        conn.close()

    # 文脈の person を AGENT_SELF へずらす。植えた行の所有者は `default`
    # （010 が入れた列の既定値）なので、所有者絞りが残っていれば 0 件になる。
    rows = _mem().for_person(AGENT_SELF_ID)._observations.recency_fallback(50, "observation")
    summaries = {r["summary"] for r in rows}

    assert f"recency self {tag}" in summaries
    assert f"recency user {tag}" in summaries


def test_keyword_fallback_does_not_filter_by_owner() -> None:
    tag = uuid.uuid4().hex[:8]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _insert(cur, f"k-self-{tag}", f"keyword {tag} self", "observation", AGENT_SELF_ID, _NOW)
            _insert(cur, f"k-user-{tag}", f"keyword {tag} user", "observation", DEFAULT_PERSON_ID,
                    _NOW + timedelta(seconds=1))
    finally:
        conn.close()

    # 所有者絞りが残っていれば 0 件になる文脈で引く（上と同じ理由）。
    rows = _mem().for_person(AGENT_SELF_ID)._observations.keyword_fallback(tag, 50, "observation")
    summaries = {r["summary"] for r in rows}

    assert f"keyword {tag} self" in summaries
    assert f"keyword {tag} user" in summaries
