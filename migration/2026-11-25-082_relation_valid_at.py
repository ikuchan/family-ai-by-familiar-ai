"""関係が成り立った時 `relations.valid_at` を足す（記-ah・2026-11-25・本人の承認）。

記録が語る事実が入れ替わったとき、古い記録は隠さず、種類「置き換え」の関係で結ぶ（`設計方針_事実の移り変わり`）。
置き換えた事実がいつから成り立ったか（文の中の「先月から」を日付に直した時。古い事実はこの時まで）を、この列に置く。
`created_at` は関係を書いた時刻で、成り立った時とは違う。

今ある関係は空のまま（NULL）。新しい表は作らない。冪等（`IF NOT EXISTS`）。src を import せず自前完結させる。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE relations ADD COLUMN IF NOT EXISTS valid_at timestamptz")
