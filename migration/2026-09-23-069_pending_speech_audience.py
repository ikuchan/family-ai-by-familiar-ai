"""保留に**宛先の条件**を持たせる（出-ap・2026-09-23・`イベント駆動ループ`）。

話したいことは、それを言うのに誰が要るかで 4 段に分かれる（0 誰もいなくても／1 誰かいたら／
2 家族がいたら／3 特定の誰かがいたら）。段 3 は既にある `target_person_id` が表す。この列は
0〜2 を持つ。

既定は **1（誰かいたら）**——いままでの振る舞いと同じ。既存の行もこれになる。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE pending_speech ADD COLUMN IF NOT EXISTS audience "
            "integer NOT NULL DEFAULT 1"
        )
