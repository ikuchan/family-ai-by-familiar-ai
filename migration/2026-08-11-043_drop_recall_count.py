"""`observations.recall_count` を落とす（situated V2 の前段）。

017 が入れた列で、実効半減期を `base_half_life × 2^recall_count` で伸ばす
**強化A（durability）**のためのものだった。`課題5` F 節が「半減期延長は 根づき の n と
役割重複」として廃止を確定させている（重要度は n が担い、t は純粋な時間減衰のみ）。

採点側は既に使っていない。`_score_breakdown` は引数で受け取るだけで `DecayState` へ
渡しておらず、残っていたのは SELECT 3箇所・UPDATE 2箇所・引数の受け渡しだけである。
2026-08-03 のダンプでは 5080 行のうち 637 行が `recall_count≠0` だったが、その値は
順位に一切効いていなかった。

強化A が生きていた頃は、`recall_count` が 20 なら半減期が 3×2^20 日＝8600年になり、
何度も想起された古い記録が永久に t=1 になっていた（実機で 47日前の挨拶が t=1.000 で
上位を占め、5秒前の自分の発話を押し出した）。

若返り（時間の起点 `last_recalled_at` の更新）は `apply_verdicts` が担い続ける。
`last_recalled_at` はこの段では動かさない（044 で `situated_memories` へ移す）。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS recall_count")
