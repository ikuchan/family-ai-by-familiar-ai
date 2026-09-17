"""タイマーの一時停止・再開（知-o 段 4・2026-09-18・`設計方針_タイマー` v0.5 §12）。

何度止めても再開しても合うように 2 列で持つ：`paused_at`＝いま止めている時刻（動いていれば NULL）、
`paused_total_sec`＝止めていた累計（再開のたびに `now − paused_at` を足し、`due` を同じだけ伸ばす）。
残り＝`due + 累計 − now`（止めている間は `due + 累計 − paused_at`）。鳴らす側（`due_now`）は
`paused_at IS NOT NULL` を拾わない。`listen`＝聞かない設定の中で、そのタイマーに限り聞くようにした印
（GUI の `/mic on`・段 5）。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE timers ADD COLUMN IF NOT EXISTS paused_at timestamptz")
        cur.execute(
            "ALTER TABLE timers ADD COLUMN IF NOT EXISTS paused_total_sec "
            "double precision NOT NULL DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE timers ADD COLUMN IF NOT EXISTS listen boolean NOT NULL DEFAULT false"
        )
