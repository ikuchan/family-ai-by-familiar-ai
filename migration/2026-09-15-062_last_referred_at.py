"""面に**参照された時**（`last_referred_at`）を持たせる（記-a-ろ-ろ・2026-09-15）。

思い出した時（`last_recalled_at`・W に載った）とは別の時刻である。主LLM が申告で
`important`／`referred` と言った時に `apply_verdicts` が記す。REST 内省の層 1 が「前回の内省
以降に参照されなかった核」を選んで根づき $n$ を $\Delta$ 減らすための鍵（`出来事を畳む` §4）。
$n$ 自体からは分からない——`referred` は $n$ を動かさず、前回の $n$ も残していない。

計測ログの `申告` 行にも同じ id が載るが、記録の状態は DB に置く（ログは復元の材料にしない）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE situated_memories ADD COLUMN IF NOT EXISTS last_referred_at TIMESTAMPTZ"
        )
