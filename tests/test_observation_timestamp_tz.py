"""Tests for writing observation timestamps on the DB clock（時刻の9時間ずれ）.

`observations.timestamp` は tz を持たない `datetime.now()` で書かれていた。列は
`timestamp with time zone` なので、JST の壁掛け時計の値がそのまま UTC として
解釈され、実時刻より9時間先に保存されていた。

同じ表の `last_recalled_at` は SQL の `now()`（DB は UTC）で書かれるため、2つの
時刻列が別の時計を指す。t 軸の起点は「last_recalled_at があればそれ、無ければ
timestamp」なので、**想起した瞬間に起点が9時間後退する**＝強化B（想起で新しさが
若返る）と逆の動きになっていた。
"""

from __future__ import annotations

import os

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = os.environ["DATABASE_URL"]


def _pg():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory(person_id=DEFAULT_PERSON_ID)


def test_saved_timestamp_matches_the_db_clock() -> None:
    """保存した観測の timestamp が DB の現在時刻と一致する（9時間ずれない）。"""
    content = f"tz check {uuid.uuid4()}"
    with patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1] * 1024]):
        _mem().save(content, direction="会話", kind="conversation")

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp, now() AS db_now FROM observations WHERE content = %s",
            (content,),
        )
        row = cur.fetchone()
    conn.close()

    assert row is not None, "観測が保存されていない"
    drift = abs((row["timestamp"] - row["db_now"]).total_seconds())
    assert drift < 300, f"DB の時計と {drift / 3600:.1f} 時間ずれている"


def test_saved_timestamp_is_not_in_the_future() -> None:
    """未来の時刻で保存されない（反証側：ずれていれば9時間先になる）。"""
    content = f"tz future check {uuid.uuid4()}"
    with patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1] * 1024]):
        _mem().save(content, direction="会話", kind="conversation")

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp > now() + interval '1 minute' AS in_future "
            "FROM observations WHERE content = %s",
            (content,),
        )
        row = cur.fetchone()
    conn.close()

    assert row is not None and not row["in_future"], "未来の時刻で保存されている"


def test_override_date_is_stored_as_local_end_of_day() -> None:
    """日付指定の保存も DB の時計に載る（その日の 23:59:59 JST 相当）。"""
    content = f"tz override {uuid.uuid4()}"
    with patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1] * 1024]):
        _mem().save(
            content, direction="会話", kind="day_summary", override_date="2026-07-01"
        )

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp FROM observations WHERE content = %s", (content,)
        )
        row = cur.fetchone()
    conn.close()

    assert row is not None
    # 2026-07-01 23:59:59 JST == 2026-07-01 14:59:59 UTC
    expected = datetime(2026, 7, 1, 14, 59, 59, tzinfo=timezone.utc)
    assert abs(row["timestamp"] - expected) < timedelta(seconds=2), row["timestamp"]
