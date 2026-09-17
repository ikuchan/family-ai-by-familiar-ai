"""アラーム（知-q・2026-09-18・`設計方針_アラーム` v0.1）。タイマーとは別物なので別の表。

遠い時刻に起こす・知らせる。`at`（timestamptz）に鳴る。`passes_quiet` は静穏時間に鳴らしてよいと
本人が確かめた印。O には `direction='予定'` の記録を別に書く（タイマーと同じ・共有するのは基盤だけ）。
`timers` に残っていた `at` 由来の行は区別できず、未来の未発火は無いので移さない。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alarms (
                id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                label        text NOT NULL,
                at           timestamptz NOT NULL,
                set_at       timestamptz NOT NULL DEFAULT now(),
                fired_at     timestamptz,
                cancelled_at timestamptz,
                asked_by     text NOT NULL DEFAULT '',
                obs_id       text,
                passes_quiet boolean NOT NULL DEFAULT false
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_alarms_at ON alarms(at) "
            "WHERE fired_at IS NULL AND cancelled_at IS NULL"
        )
