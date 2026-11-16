"""日付だけの記録は、その日の終わりがまだ来ていなければ、書いた時刻で記録する（記-aa・2026-11-16・本人の決定）。

日付だけを渡した記録（`override_date`）は、その日のローカル 23:59:59 で記録する。日のまとめ（`rest_fold`）と
まとめ（`rest_core`）は、材料に当日の行が入ると、まだ来ていない時刻を持ち、W の日時の欄に先の時刻が出た
（出-ds の見直しで見つけた）。日付を時刻に変える 1 か所（`materialize_save_event`）で、その日の終わりと
いまの早いほうを取る。前の日以前のぶんは、いままでどおりその日の終わりで記録する
（`test_observation_timestamp_tz.py::test_override_date_is_stored_as_local_end_of_day`）。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store import clock
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


def _stored_gap(override_date: str) -> float:
    """保存した行の時刻と DB のいまの差（秒）。"""
    content = f"記-aa {uuid.uuid4()}"
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1] * 1024]),
    ):
        ObservationMemory(person_id=DEFAULT_PERSON_ID).save_with_id(
            content, direction="記憶", override_date=override_date
        )
    conn = psycopg2.connect(
        os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXTRACT(EPOCH FROM (timestamp - now())) AS gap FROM observations WHERE content = %s",
            (content,),
        )
        row = cur.fetchone()
    conn.close()
    assert row is not None, "観測が保存されていない"
    return float(row["gap"])


def test_a_summary_of_today_is_written_at_the_write_time():
    """今日の日付なら、まだ来ていない 23:59:59 ではなく、書いた時刻で記録する。"""
    today = datetime.now(clock.local_tz()).strftime("%Y-%m-%d")
    assert abs(_stored_gap(today)) < 60


def test_a_future_date_is_also_written_at_the_write_time():
    """明日の日付（タイムゾーンのずれなどで先の日付が来たとき）も、先の時刻にしない。"""
    tomorrow = (datetime.now(clock.local_tz()) + timedelta(days=1)).strftime("%Y-%m-%d")
    assert abs(_stored_gap(tomorrow)) < 60
