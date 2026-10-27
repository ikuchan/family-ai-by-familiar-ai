"""記録の種類（`kind`）の列と索引を落とし、向き（`direction`）に索引を足す（記-u 段 4・2026-10-27・本人の決定）。

「何の記録か」は向きだけで持つ（記-u・案ウ。`MI データモデル` 付録A の「kind 廃止（案A）」）。種類にしか無かった
情報は 077 で向きへ移した。読み手は段 2 で向きへ、書く口は段 3 で OIF へ移してある。

向きの索引は、向きで引く読み手（記憶の木の要約・最終版・畳み込みの材料）のため。本番は 2026-10-27 に 21,921 行。

冪等（`IF EXISTS`／`IF NOT EXISTS`）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_obs_kind")
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS kind")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_direction ON observations(direction)")
