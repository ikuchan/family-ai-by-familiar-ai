"""ストップウォッチ（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。タイマーとは別物なので別の表。

「今から測って」で始め、「止めて」で止め、経過を答える。鳴らない。`started_at` から `stopped_at` までが
測った長さ。`expired` は寿命（`STOPWATCH_MAX_SEC`・6 時間）で T が自動で止めた印。O には `direction='予定'`
の記録を別に書く（タイマー・アラームと同じ）。`timers` に残っていた `due` が NULL の行（2 本・止め済み）は
移さない。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stopwatches (
                id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                label      text NOT NULL,
                started_at timestamptz NOT NULL DEFAULT now(),
                stopped_at timestamptz,
                asked_by   text NOT NULL DEFAULT '',
                obs_id     text,
                expired    boolean NOT NULL DEFAULT false
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_stopwatches_running ON stopwatches(started_at) "
            "WHERE stopped_at IS NULL"
        )
