"""タイマー（知-n・2026-09-15・`設計方針_タイマー` v0.1）。

アラーム・タイマー・ストップウォッチの状態を**列で**持つ（content の時刻を読まない）。
再起動をまたいで残る。O には `direction='予定'` の記録を別に書き、想起に載る（「約束した」が記憶になる）。
`due` が NULL ならストップウォッチ（鳴らない・経過を答えるだけ）。`passes_quiet` は静穏時間・沈黙の依頼を
通り抜ける（頼んだ本人が確かめたうえで登録した）印。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS timers (
                id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                label        text NOT NULL,
                due          timestamptz,
                started_at   timestamptz NOT NULL DEFAULT now(),
                fired_at     timestamptz,
                cancelled_at timestamptz,
                asked_by     text NOT NULL DEFAULT '',
                obs_id       text,
                passes_quiet boolean NOT NULL DEFAULT false
            )
            """
        )
        # T が毎 tick「鳴らす頃合いの未発火・未取消」を引く道。
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_timers_due ON timers(due) "
            "WHERE fired_at IS NULL AND cancelled_at IS NULL"
        )
