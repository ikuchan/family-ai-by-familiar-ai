"""Tests for append-only memory event log and pending job queue (PostgreSQL)."""

from __future__ import annotations

import os

import json
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


_DB_URL = os.environ["DATABASE_URL"]


def _pg_conn():
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = False
    return conn


def test_save_appends_event_and_pending_job() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save_with_id("hello world", direction="会話", emotion="curious")[1]

    conn = _pg_conn()
    # 落ちても接続を閉じる。開いたまま残ると、次の試験の後片づけ（TRUNCATE）がその錠を待って止まる。
    try:
        _check_the_event(conn)
    finally:
        conn.close()


def _check_the_event(conn) -> None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT event_id, event_type, payload_json FROM memory_events")
        event = cur.fetchone()
        assert event is not None
        assert event["event_type"] == "memory.save"

        payload = json.loads(event["payload_json"])
        assert payload["content"] == "hello world"
        assert payload["direction"] == "会話"
        assert "kind" not in payload  # 記録の種類は持たない（記-u 段 4）
        assert payload["emotion"] == "curious"

        cur.execute(
            "SELECT job_type, status, attempts FROM memory_jobs WHERE event_id = %s",
            (event["event_id"],),
        )
        job = cur.fetchone()
        assert job is not None
        assert job["job_type"] == "materialize_observation"
        assert job["status"] == "pending"
        assert job["attempts"] == 0


def test_same_content_within_the_window_is_written_once() -> None:
    """040 で鍵を落としたあと、重複防止は時間窓が担う。

    鍵（`memory_events.dedupe_key`）はキューへ積む段で弾いていた。時間窓は
    `observations` へ書く段で弾く。掛かる場所が違うので、`memory_events` は2件に
    なるが、**記憶そのものは1件に収まる**。
    """
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save_with_id("same payload", direction="会話")[1]
        assert mem.save_with_id("same payload", direction="会話")[1]

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM observations")
            assert cur.fetchone()[0] == 1, "時間窓が二度目を弾けていない"
    finally:
        conn.close()


def test_save_can_enqueue_without_immediate_materialization() -> None:
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1, 0.2, 0.3]]),
    ):
        mem = ObservationMemory()
        assert mem.save_with_id("queued only", direction="会話", materialize_now=False)[1]

    conn = _pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM memory_events")
            assert cur.fetchone()["cnt"] == 1
            cur.execute("SELECT status FROM memory_jobs")
            job_row = cur.fetchone()
            assert job_row is not None
            assert job_row["status"] == "pending"
            cur.execute("SELECT COUNT(*) AS cnt FROM observations")
            assert cur.fetchone()["cnt"] == 0
    finally:
        conn.close()
